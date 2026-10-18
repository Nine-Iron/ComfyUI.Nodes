// moFlo downloads the models you have liked on Civitai into a ComfyUI
// models folder, without ever fetching one you already have.
//
// Files are saved with their base model in the name, where -tag says:
// front flux1_dev_x.safetensors, end x_flux1_dev.safetensors, dot
// x.flux1_dev.safetensors, folder loras/flux1_dev/x.safetensors, or none.
// Files saved under any of these forms are recognised, so -tag can change
// later (-rename-only moves existing files to the new form).
//
// Config: ~/.config/moflo/config.json (see below). Flags on the command line
// win for that run; -set-conf saves them as the new defaults.
// A liked model with versions for several base models gets one file per
// selected base model (the newest version for each).
//
// Before anything is downloaded, each one is checked against the WHOLE models
// tree, whatever folder a file was filed under:
//
//  1. by its tagged name, and by Civitai's untagged name, for every version;
//  2. by size, then SHA-256, to catch a copy saved under any other name.
//
// Only what is missing is downloaded. Nothing is ever overwritten.
//
// Setup: create an API key at https://civitai.com/user/account and either
// export CIVITAI_TOKEN=... or put it in ~/.config/civitai/token.
//
// Previews: each model file gets the first still image from its version's
// Civitai page saved beside it as <name>.preview.<ext>, which ComfyUI's Model
// Library shows. -previews N keeps extras as <name>.preview-2.<ext>, ... to
// choose from: rename the one you want to <name>.preview.<ext>. Files you
// already have get previews too, unless one is already there.
// Beside each preview goes <name>.preview.json (or .preview-N.json) with what
// Civitai records about how that picture was made (prompt, sampler, seed...),
// which the Model Preview Popup shows. -preview-info fills these in for
// previews saved before, by matching them to Civitai's images byte for byte.
//
// Checksums: only of what it downloads (to verify it), with -checksum, and with
// -local. The tool keeps a state file in the models folder
// (.moflo.json) with what it has seen; -checksum checksums every model
// file with none on record (each file once), and the recorded checksums are
// used to spot copies saved under other names and to tell apart files that
// share a name and size. Without them such cases are left alone, never guessed.
//
// Usage:
//
//	moflo -n                 show what would happen, change nothing
//	moflo                    show the plan, ask, then download
//	moflo -yes               no question asked
//	moflo -base pony         only Pony models, this run
//	moflo -base sdxl1,sd1    only these base models, this run
//	moflo -reconfig          pick again which base models are downloaded
//	moflo -set-conf -previews 3   save flags as the defaults in the config file
//	moflo -rename            also rename files you already have to the tagged form
//	moflo -rename-only       only that: rename, no downloads or previews
//	moflo -rename-only -local   also files that are not liked models, looked up by checksum
//	moflo -rename-only -local -guess   and tag the ones Civitai does not know from their metadata
//	moflo -previews 4        keep up to 4 preview images per model to choose from
//	moflo -previews-only -preview-info   save generation settings for previews already on disk
//	moflo -checksum          checksum model files with none on record (slow, once per file)
//	moflo -latest            also fetch the newest version when you hold an older one
//
// The config file (~/.config/moflo/config.json) holds every flag's
// default, which base models are downloaded (asked on the first run, and for
// new ones), per-base previews and latest, and per base model the liked models
// with what became of each; "download": false on one keeps it from being
// downloaded for that base. -base picks the base models for one run instead.
//
// Build: go build -o moflo moflo.go   (standard library only)
package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"math"
	"mime"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unicode"
)

const (
	version       = "2026.10.18-3" // bumped with every build handed out
	userAgent     = "moflo/" + version
	sizeSlack     = 1024             // bytes; the API reports sizes in KB
	stallTimeout  = 2 * time.Minute  // abort a download that stops sending data
	apiTimeout    = 60 * time.Second // per API request
	maxAttempts   = 4                // per file / per API request
	spaceHeadroom = 2 << 30          // keep 2 GB free on the disk
)

// Civitai model type -> folder under the ComfyUI models directory.
var typeDirs = map[string]string{
	"Checkpoint":       "checkpoints",
	"LORA":             "loras",
	"LoCon":            "loras",
	"DoRA":             "loras",
	"TextualInversion": "embeddings",
	"VAE":              "vae",
	"Controlnet":       "controlnet",
	"Upscaler":         "upscale_models",
	"Hypernetwork":     "hypernetworks",
	"MotionModule":     "animatediff_models",
}

// ---------------------------------------------------------------- API types

type apiFile struct {
	Name        string         `json:"name"`
	Type        string         `json:"type"`
	SizeKB      float64        `json:"sizeKB"`
	Primary     bool           `json:"primary"`
	Hashes      map[string]any `json:"hashes"`
	DownloadURL string         `json:"downloadUrl"`
}

func (f apiFile) bytes() int64 { return int64(math.Round(f.SizeKB * 1024)) }

func (f apiFile) sha256() string {
	for k, v := range f.Hashes {
		if s, ok := v.(string); ok && strings.EqualFold(k, "SHA256") {
			return strings.ToLower(strings.TrimSpace(s))
		}
	}
	return ""
}

type apiVersion struct {
	ID                int64      `json:"id"`
	Name              string     `json:"name"`
	BaseModel         string     `json:"baseModel"`
	Availability      string     `json:"availability"`
	EarlyAccessEndsAt string     `json:"earlyAccessEndsAt"`
	UsageControl      string     `json:"usageControl"`
	Files             []apiFile  `json:"files"`
	Images            []apiImage `json:"images"`
}

type apiModel struct {
	ID            int64        `json:"id"`
	Name          string       `json:"name"`
	Type          string       `json:"type"`
	Mode          string       `json:"mode"`
	ModelVersions []apiVersion `json:"modelVersions"`
	Creator       struct {
		Username string `json:"username"`
	} `json:"creator"`
}

type apiPage struct {
	Items    []apiModel `json:"items"`
	Metadata struct {
		NextCursor json.RawMessage `json:"nextCursor"`
	} `json:"metadata"`
}

type httpError struct {
	status int
	retry  time.Duration
}

func (e *httpError) Error() string {
	switch e.status {
	case http.StatusUnauthorized:
		return "401: Civitai refused access with this API token"
	case http.StatusForbidden:
		return "403: this account is not allowed to download it (early access or restricted)"
	case http.StatusNotFound:
		return "404: not found on Civitai (removed or unpublished)"
	case http.StatusTooManyRequests:
		return "429: rate limited by Civitai"
	}
	return fmt.Sprintf("HTTP %d from Civitai", e.status)
}

func (e *httpError) retryable() bool {
	return e.status == http.StatusTooManyRequests || e.status >= 500
}

// ------------------------------------------------------------------ client

type client struct {
	base  *url.URL
	token string
	http  *http.Client
	metas map[int64]map[string]json.RawMessage // version id -> image key -> generation data
}

func (c *client) do(ctx context.Context, rawURL string, hdr map[string]string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	// The token only ever goes to the API host. Go drops this header by itself
	// when a download redirects to the storage host.
	if strings.EqualFold(req.URL.Host, c.base.Host) {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	req.Header.Set("User-Agent", userAgent)
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	return c.http.Do(req)
}

func statusError(resp *http.Response) *httpError {
	e := &httpError{status: resp.StatusCode}
	if s, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil && s > 0 && s <= 600 {
		e.retry = time.Duration(s) * time.Second
	}
	return e
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func backoff(attempt int, err error) time.Duration {
	var he *httpError
	if errors.As(err, &he) && he.retry > 0 {
		return he.retry
	}
	return time.Duration(attempt*attempt) * 3 * time.Second
}

// getJSON fetches an API URL, retrying rate limits and server errors.
func (c *client) getJSON(ctx context.Context, rawURL string, out any) error {
	var last error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		err := func() error {
			rctx, cancel := context.WithTimeout(ctx, apiTimeout)
			defer cancel()
			resp, err := c.do(rctx, rawURL, map[string]string{"Accept": "application/json"})
			if err != nil {
				return err
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<16))
				return statusError(resp)
			}
			return json.NewDecoder(resp.Body).Decode(out)
		}()
		if err == nil {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		last = err
		var he *httpError
		if errors.As(err, &he) && !he.retryable() {
			return err
		}
		if attempt < maxAttempts {
			if err := sleepCtx(ctx, backoff(attempt, err)); err != nil {
				return err
			}
		}
	}
	return last
}

// whoami checks the token and returns the account name when the API gives one.
func (c *client) whoami(ctx context.Context) (string, error) {
	var me map[string]any
	err := c.getJSON(ctx, c.base.String()+"/api/v1/me", &me)
	if err != nil {
		var he *httpError
		if errors.As(err, &he) && (he.status == http.StatusUnauthorized || he.status == http.StatusForbidden) {
			return "", err
		}
		return "", nil // endpoint unavailable; the likes request will tell
	}
	if name, ok := me["username"].(string); ok {
		return name, nil
	}
	return "", nil
}

// likedModels pages through the account's liked models.
func (c *client) likedModels(ctx context.Context, limit int) ([]apiModel, error) {
	q := url.Values{"favorites": {"true"}, "nsfw": {"true"}, "limit": {"100"}}
	models, err := c.pageModels(ctx, q, limit)
	if err != nil {
		return nil, err
	}
	// Early access versions may only be listed on request. Ask again with
	// them included and merge, so either reading of the parameter works.
	q.Set("earlyAccess", "true")
	extra, err := c.pageModels(ctx, q, limit)
	if err != nil {
		fmt.Fprintln(os.Stderr, "warning: could not list early access versions:", err)
		return models, nil
	}
	byID := map[int64]int{}
	for i, m := range models {
		byID[m.ID] = i
	}
	for _, m := range extra {
		i, ok := byID[m.ID]
		if !ok {
			if len(models) >= limit {
				continue
			}
			byID[m.ID] = len(models)
			models = append(models, m)
			continue
		}
		have := map[int64]bool{}
		for _, v := range models[i].ModelVersions {
			have[v.ID] = true
		}
		added := false
		for _, v := range m.ModelVersions {
			if !have[v.ID] {
				models[i].ModelVersions = append(models[i].ModelVersions, v)
				added = true
			}
		}
		if added { // keep newest first; version ids only ever grow
			sort.SliceStable(models[i].ModelVersions, func(a, b int) bool {
				return models[i].ModelVersions[a].ID > models[i].ModelVersions[b].ID
			})
		}
	}
	return models, nil
}

// pageModels follows the cursor through every page of a models query.
func (c *client) pageModels(ctx context.Context, q url.Values, limit int) ([]apiModel, error) {
	var models []apiModel
	seenID := map[int64]bool{}
	seenCursor := map[string]bool{}
	for {
		var page apiPage
		if err := c.getJSON(ctx, c.base.String()+"/api/v1/models?"+q.Encode(), &page); err != nil {
			return nil, err
		}
		for _, m := range page.Items {
			if m.ID == 0 || !seenID[m.ID] {
				seenID[m.ID] = true
				models = append(models, m)
			}
		}
		if len(models) > limit {
			return nil, fmt.Errorf("more than %d models came back; that does not look like a likes list, "+
				"so nothing was done (raise -max if it really is)", limit)
		}
		cursor := strings.TrimSpace(string(page.Metadata.NextCursor))
		if strings.HasPrefix(cursor, `"`) {
			var s string
			if json.Unmarshal(page.Metadata.NextCursor, &s) == nil {
				cursor = s
			}
		}
		if cursor == "" || cursor == "null" || seenCursor[cursor] || len(page.Items) == 0 {
			return models, nil
		}
		seenCursor[cursor] = true
		q.Set("cursor", cursor)
	}
}

// ------------------------------------------------------------- local index

// Extensions of files that are models (and so get checksummed and matched by size).
var modelExts = map[string]bool{".safetensors": true, ".sft": true, ".ckpt": true, ".pt": true,
	".pth": true, ".bin": true, ".gguf": true, ".onnx": true}

// Image types ComfyUI shows as a model preview.
var imageExts = []string{"png", "jpg", "jpeg", "webp", "gif"}

type localFile struct {
	path  string
	size  int64
	mtime int64 // ns
	ctime int64 // ns; a move or a copy that keeps the old mtime still updates this
}

type index struct {
	root      string
	byName    map[string][]localFile // lower-case file name -> files
	byHash    map[string][]localFile // sha256 -> files, from the state file
	shaByPath map[string]string
	all       []localFile
	bytes     int64
}

func isModelFile(path string) bool { return modelExts[strings.ToLower(filepath.Ext(path))] }

// buildIndex lists every file under root, following symlinked folders.
// Files whose path is in skip (the tool's own state file) are left out.
func buildIndex(root string, skip map[string]bool) (*index, error) {
	idx := &index{root: root, byName: map[string][]localFile{}, byHash: map[string][]localFile{},
		shaByPath: map[string]string{}}
	visited := map[string]bool{}
	var walk func(dir string) error
	walk = func(dir string) error {
		real, err := filepath.EvalSymlinks(dir)
		if err != nil || visited[real] {
			return nil // broken link, or a folder already covered
		}
		visited[real] = true
		entries, err := os.ReadDir(dir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: cannot read %s: %v\n", dir, err)
			return nil
		}
		for _, e := range entries {
			p := filepath.Join(dir, e.Name())
			if skip[p] {
				continue
			}
			info, err := os.Stat(p) // follows symlinks
			if err != nil {
				continue
			}
			if info.IsDir() {
				if err := walk(p); err != nil {
					return err
				}
			} else if info.Mode().IsRegular() {
				f := localFile{path: p, size: info.Size(), mtime: info.ModTime().UnixNano()}
				f.ctime = f.mtime
				if st, ok := info.Sys().(*syscall.Stat_t); ok {
					f.ctime = int64(st.Ctim.Sec)*1e9 + int64(st.Ctim.Nsec)
				}
				idx.add(f)
			}
		}
		return nil
	}
	if info, err := os.Stat(root); err != nil || !info.IsDir() {
		return nil, fmt.Errorf("models folder %s not found", root)
	}
	return idx, walk(root)
}

func (idx *index) add(f localFile) {
	key := strings.ToLower(filepath.Base(f.path))
	idx.byName[key] = append(idx.byName[key], f)
	idx.all = append(idx.all, f)
	idx.bytes += f.size
}

func (idx *index) rel(path string) string {
	if r, err := filepath.Rel(idx.root, path); err == nil {
		return r
	}
	return path
}

func hashFile(ctx context.Context, path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	buf := make([]byte, 4<<20)
	for {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		n, err := f.Read(buf)
		h.Write(buf[:n])
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func sizeClose(local, remote int64) bool { return absDiff(local, remote) <= sizeSlack }

// sizeExact allows only for the rounding in Civitai's KB figure.
func sizeExact(local, remote int64) bool { return absDiff(local, remote) <= 16 }

func absDiff(a, b int64) int64 {
	if a > b {
		return a - b
	}
	return b - a
}

// ------------------------------------------------------------------- state

// fileState is what the tool remembers about one model file.
type fileState struct {
	Size     int64    `json:"size"`
	MTime    int64    `json:"mtime"`
	SHA256   string   `json:"sha256,omitempty"`
	Previews []string `json:"previews,omitempty"` // image URLs already saved beside it
	Base     string   `json:"base,omitempty"`     // base model tag, e.g. flux1_dev
	BaseName string   `json:"baseName,omitempty"` // what Civitai calls it, e.g. "Flux.1 D"
	Model    string   `json:"model,omitempty"`    // Civitai model name
	Title    string   `json:"title,omitempty"`    // the model name without the base model list at its end (shown in ComfyUI)
	Version  string   `json:"version,omitempty"`  // Civitai version name
	// What -local learned about a file no liked model accounts for.
	ModelType    string `json:"modelType,omitempty"`
	VersionName  string `json:"versionName,omitempty"`
	VersionID    int64  `json:"versionId,omitempty"`
	NotOnCivitai bool   `json:"notOnCivitai,omitempty"`
}

// setBase notes which base model a file is for (read by the preview popup).
func (s *stateFile) setBase(idx *index, path string, it item, baseName string) {
	e := s.Files[idx.rel(path)]
	if e == nil {
		return
	}
	if it.model.Name != "" { // the name ComfyUI shows for it (read by the preview popup)
		e.Title, e.Version = cleanTitle(it.model.Name), it.ver.Name
	}
	if it.tag == "" {
		return
	}
	e.Base, e.BaseName, e.Model = it.tag, baseName, it.model.Name
	e.NotOnCivitai = false
}

// stateFile is the tool's memory between runs, kept in the models folder.
type stateFile struct {
	Version  int                   `json:"version"`
	LastScan time.Time             `json:"lastScan"`
	Files    map[string]*fileState `json:"files"` // by path relative to the models folder
	// Early access versions Civitai refused to hand over, by version id.
	EarlyAccess map[string]*eaRecord `json:"earlyAccess,omitempty"`
	path        string
	existed     bool
}

func loadState(path string) (*stateFile, error) {
	st := &stateFile{Version: 1, Files: map[string]*fileState{}, EarlyAccess: map[string]*eaRecord{}, path: path}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return st, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(data, st); err != nil {
		return nil, fmt.Errorf("%s is damaged (%v); move it aside and the next run starts fresh", path, err)
	}
	if st.Files == nil {
		st.Files = map[string]*fileState{}
	}
	if st.EarlyAccess == nil {
		st.EarlyAccess = map[string]*eaRecord{}
	}
	st.existed = true
	return st, nil
}

func (s *stateFile) save() error {
	data, err := json.MarshalIndent(s, "", " ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

func (s *stateFile) saveOrWarn() {
	if err := s.save(); err != nil {
		fmt.Fprintln(os.Stderr, "warning: could not save", s.path+":", err)
	}
}

// record notes a model file the tool has just put in place.
func (s *stateFile) record(idx *index, path, sha string) {
	info, err := os.Stat(path)
	if err != nil {
		return
	}
	rel := idx.rel(path)
	e := s.Files[rel]
	if e == nil {
		e = &fileState{}
		s.Files[rel] = e
	}
	e.Size, e.MTime = info.Size(), info.ModTime().UnixNano()
	if sha != "" {
		e.SHA256 = sha
	}
}

// syncState lines the state up with the files on disk, then checksums the
// model files that need it: by default only those new or changed since the
// last run, and every one not yet checksummed when all is set. The first run
// only records what is there.
func syncState(ctx context.Context, idx *index, st *stateFile, all bool) error {
	seen := map[string]bool{}
	var todo []localFile
	var todoBytes int64
	for _, l := range idx.all {
		if !isModelFile(l.path) {
			continue
		}
		rel := idx.rel(l.path)
		seen[rel] = true
		e := st.Files[rel]
		if e == nil || e.Size != l.size || e.MTime != l.mtime {
			ne := &fileState{Size: l.size, MTime: l.mtime}
			if e != nil { // a changed file keeps what is known about it, not its checksum
				ne.Previews, ne.Base, ne.BaseName, ne.Model = e.Previews, e.Base, e.BaseName, e.Model
				ne.Title, ne.Version = e.Title, e.Version
			}
			e = ne
			st.Files[rel] = e
		}
		if e.SHA256 == "" && all {
			todo = append(todo, l)
			todoBytes += l.size
		}
	}
	for rel := range st.Files {
		if !seen[rel] {
			delete(st.Files, rel)
		}
	}
	if len(todo) > 0 {
		fmt.Fprintf(os.Stderr, "Checksumming %d model file(s) with no checksum on record, %s (-checksum)...\n", len(todo), human(todoBytes))
	}
	for i, l := range todo {
		fmt.Fprintf(os.Stderr, "  [%d/%d] %s (%s)\n", i+1, len(todo), idx.rel(l.path), human(l.size))
		sum, err := hashFile(ctx, l.path)
		if err != nil {
			if ctx.Err() != nil {
				st.saveOrWarn() // keep the checksums done so far
				return ctx.Err()
			}
			fmt.Fprintf(os.Stderr, "  warning: %v\n", err)
			continue
		}
		st.Files[idx.rel(l.path)].SHA256 = sum
		st.saveOrWarn()
	}
	for _, l := range idx.all {
		if e := st.Files[idx.rel(l.path)]; e != nil && e.SHA256 != "" {
			idx.byHash[e.SHA256] = append(idx.byHash[e.SHA256], l)
			idx.shaByPath[l.path] = e.SHA256
		}
	}
	return nil
}

// ------------------------------------------------------------- one run only

// lockModels makes sure only one run works on a models folder at a time: two
// runs planning the same downloads fetch everything twice. The lock is the
// kernel's (flock), so it goes away by itself when a run ends or dies.
func lockModels(root string) (*os.File, error) {
	path := filepath.Join(root, ".moflo.lock")
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		who, _ := os.ReadFile(path)
		f.Close()
		msg := strings.TrimSpace(string(who))
		if msg == "" {
			msg = "another run"
		}
		return nil, fmt.Errorf("moFlo is already running on %s (%s); let it finish, or stop it first", root, msg)
	}
	f.Truncate(0)
	fmt.Fprintf(f, "pid %d, started %s", os.Getpid(), time.Now().Format("Jan 2 15:04"))
	f.Sync()
	return f, nil
}

// ----------------------------------------------------- copies and leftovers

// copyGroup is one model file kept, and byte-identical copies of it to remove.
type copyGroup struct {
	keep localFile
	drop []localFile
}

// untagged is a file name without any base model tag the tool may have put in
// it, so the copies -tag front and -tag folder make of one file line up.
func untagged(path string) string {
	name := strings.ToLower(filepath.Base(path))
	ext := filepath.Ext(name)
	stem := strings.TrimSuffix(name, ext)
	for t := range knownTags {
		switch {
		case strings.HasPrefix(stem, t+"_"):
			stem = strings.TrimPrefix(stem, t+"_")
		case strings.HasSuffix(stem, "_"+t):
			stem = strings.TrimSuffix(stem, "_"+t)
		case strings.HasSuffix(stem, "."+t):
			stem = strings.TrimSuffix(stem, "."+t)
		}
	}
	return stem + ext
}

// topFolder is the models folder a file belongs to (loras, checkpoints, ...).
func topFolder(idx *index, path string) string {
	rel := idx.rel(path)
	if i := strings.IndexAny(rel, `/\`); i > 0 {
		return rel[:i]
	}
	return ""
}

// inTagForm reports whether a file is named the way -tag names files now.
func inTagForm(path string) bool {
	dir, stem := filepath.Base(filepath.Dir(path)), strings.ToLower(strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)))
	for t := range knownTags {
		tagged := false
		switch tagMode {
		case "folder":
			tagged = strings.EqualFold(dir, t)
		case "front":
			tagged = strings.HasPrefix(stem, t+"_")
		case "end":
			tagged = strings.HasSuffix(stem, "_"+t)
		case "dot":
			tagged = strings.HasSuffix(stem, "."+t)
		}
		if tagged {
			return true
		}
	}
	return false
}

// ------------------------------------------------- checksums moFlo wants

// hashJob is a file moFlo would like to checksum, and why.
type hashJob struct {
	l   localFile
	why string
}

// copyCandidates lists files that look like copies of each other (the same
// size in one models folder, and the same name once base model tags are taken
// off) and have no checksum on record yet.
func copyCandidates(idx *index) []hashJob {
	type key struct {
		top  string
		size int64
	}
	bySize := map[key][]localFile{}
	for _, l := range idx.all {
		if isModelFile(l.path) {
			k := key{topFolder(idx, l.path), l.size}
			bySize[k] = append(bySize[k], l)
		}
	}
	var out []hashJob
	for _, group := range bySize {
		if len(group) < 2 {
			continue
		}
		for _, l := range group {
			if idx.shaByPath[l.path] != "" {
				continue
			}
			for _, o := range group {
				if o.path != l.path && untagged(o.path) == untagged(l.path) {
					out = append(out, hashJob{l, fmt.Sprintf("looks like a copy of %s (same size, same name apart "+
						"from the base model tag); if it is, one of the two can go", idx.rel(o.path))})
					break
				}
			}
		}
	}
	return out
}

// ambiguousCandidates lists files with no checksum on record that may be a
// liked file under another name, and the files whose name and size fit more
// than one liked file (unsure: left alone unless checksummed).
func ambiguousCandidates(idx *index, models []apiModel) ([]hashJob, map[string]bool) {
	claims := map[string]map[string]bool{} // local path -> liked files that fit it by name and size
	shared := sharedNames(models)
	type liked struct {
		size  int64
		label string
	}
	var withSum []liked
	for _, m := range models {
		for _, v := range m.ModelVersions {
			f, ok := modelFile(v)
			if !ok {
				continue
			}
			tag := ""
			if !baseAgnostic[m.Type] {
				tag = baseTag(v.BaseModel)
			}
			label := m.Name
			if tag != "" {
				label += " [" + tag + "]"
			}
			if f.sha256() != "" && f.bytes() > 0 {
				withSum = append(withSum, liked{f.bytes(), label})
			}
			save, _ := saveName(m, v, f, shared)
			for _, name := range (candidate{file: f, tag: tag, save: save}).names() {
				for _, l := range idx.byName[strings.ToLower(name)] {
					if f.bytes() == 0 || sizeClose(l.size, f.bytes()) {
						if claims[l.path] == nil {
							claims[l.path] = map[string]bool{}
						}
						claims[l.path][label] = true
					}
				}
			}
		}
	}
	list := func(names []string) string {
		sort.Strings(names)
		names = slices.Compact(names)
		if len(names) > 3 {
			return strings.Join(names[:3], ", ") + fmt.Sprintf(" and %d more", len(names)-3)
		}
		return strings.Join(names, ", ")
	}
	var out []hashJob
	unsure := map[string]bool{}
	for _, l := range idx.all {
		if idx.shaByPath[l.path] != "" || !isModelFile(l.path) {
			continue
		}
		if len(claims[l.path]) > 1 {
			unsure[l.path] = true
			out = append(out, hashJob{l, fmt.Sprintf("its name and size fit %d liked files (%s); only its checksum "+
				"tells which one it is, so the others are not downloaded until then",
				len(claims[l.path]), list(slices.Collect(maps.Keys(claims[l.path]))))})
			continue
		}
		var same []string
		for _, w := range withSum {
			if sizeExact(l.size, w.size) {
				same = append(same, w.label)
			}
		}
		if len(same) > 0 {
			unsure[l.path] = true
			out = append(out, hashJob{l, fmt.Sprintf("exactly the size of liked %s; if it is one of them under "+
				"another name, that one is not downloaded again. Unchecked, those stay left alone", list(same))})
		}
	}
	return out, unsure
}

// askChecksums says which files moFlo wants to checksum and why, and asks.
// -yes answers yes; no answer to read counts as no.
func askChecksums(idx *index, jobs []hashJob, yes bool) bool {
	var total int64
	for _, j := range jobs {
		total += j.l.size
	}
	fmt.Printf("\nmoFlo would like to checksum %d file(s), %s, before planning. That reads every byte, so it\n", len(jobs), human(total))
	fmt.Println("takes a while for big files; each one is done once and remembered. Why:")
	for _, j := range jobs {
		fmt.Printf("  %s  (%s)\n      %s\n", idx.rel(j.l.path), human(j.l.size), j.why)
	}
	fmt.Println("If not: copies are not cleaned up, and liked files that may already be here are left alone")
	fmt.Println("(listed below) rather than downloaded twice. You are asked again next run.")
	if yes {
		fmt.Println("Checksumming them (-yes).")
		return true
	}
	fmt.Print("Checksum them now? [y/N] ")
	answer, err := stdin.ReadString('\n')
	if err != nil && strings.TrimSpace(answer) == "" {
		fmt.Println("\nNo answer to read: not checksummed (-yes says yes).")
		return false
	}
	a := strings.ToLower(strings.TrimSpace(answer))
	return a == "y" || a == "yes"
}

// hashJobs checksums the files and records the results in the index and state.
func hashJobs(ctx context.Context, idx *index, st *stateFile, jobs []hashJob) error {
	for i, j := range jobs {
		l := j.l
		fmt.Fprintf(os.Stderr, "  [%d/%d] %s (%s)\n", i+1, len(jobs), idx.rel(l.path), human(l.size))
		sum, err := hashFile(ctx, l.path)
		if err != nil {
			if ctx.Err() != nil {
				st.saveOrWarn()
				return ctx.Err()
			}
			fmt.Fprintf(os.Stderr, "  warning: %v\n", err)
			continue
		}
		idx.shaByPath[l.path] = sum
		idx.byHash[sum] = append(idx.byHash[sum], l)
		if e := st.Files[idx.rel(l.path)]; e != nil {
			e.SHA256 = sum
		}
		st.saveOrWarn()
	}
	return nil
}

// groupCopies finds model files stored more than once: the same bytes (by
// the checksums on record) under the same models folder, as two runs with
// different -tag leave behind. Links are never counted as copies.
func groupCopies(idx *index) []copyGroup {
	type hkey struct{ top, sha string }
	byHash := map[hkey][]localFile{}
	for _, l := range idx.all {
		if sha := idx.shaByPath[l.path]; sha != "" && isModelFile(l.path) {
			k := hkey{topFolder(idx, l.path), sha}
			byHash[k] = append(byHash[k], l)
		}
	}
	// Files that a symbolic link somewhere in the tree points to: never removed.
	targets := linkTargets(idx)
	linked := func(fi os.FileInfo) bool { return isLinkTarget(targets, fi) }
	var out []copyGroup
	for _, group := range byHash {
		var real []localFile // links (symbolic or hard) are not copies
		var infos []os.FileInfo
		for _, l := range group {
			li, err := os.Lstat(l.path)
			if err != nil || li.Mode()&os.ModeSymlink != 0 {
				continue
			}
			same := false
			for _, o := range infos {
				if os.SameFile(o, li) {
					same = true
				}
			}
			if !same {
				real = append(real, l)
				infos = append(infos, li)
			}
		}
		if len(real) < 2 {
			continue
		}
		isLinked := map[string]bool{}
		for i, l := range real {
			isLinked[l.path] = linked(infos[i])
		}
		// Keep one a link points to, then the one named the way -tag names
		// files now, then the one with the most previews and info files, then
		// the oldest. Every copy a link points to stays.
		sort.SliceStable(real, func(i, j int) bool {
			a, b := real[i], real[j]
			if la, lb := isLinked[a.path], isLinked[b.path]; la != lb {
				return la
			}
			if ta, tb := inTagForm(a.path), inTagForm(b.path); ta != tb {
				return ta
			}
			if na, nb := len(sidecars(a.path)), len(sidecars(b.path)); na != nb {
				return na > nb
			}
			if a.mtime != b.mtime {
				return a.mtime < b.mtime
			}
			return a.path < b.path
		})
		g := copyGroup{keep: real[0]}
		for _, l := range real[1:] {
			if !isLinked[l.path] {
				g.drop = append(g.drop, l)
			}
		}
		if len(g.drop) > 0 {
			out = append(out, g)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].keep.path < out[j].keep.path })
	return out
}

// linkTargets lists the files symbolic links in the tree point to.
func linkTargets(idx *index) []os.FileInfo {
	var out []os.FileInfo
	for _, l := range idx.all {
		if li, err := os.Lstat(l.path); err == nil && li.Mode()&os.ModeSymlink != 0 {
			if ti, err := os.Stat(l.path); err == nil {
				out = append(out, ti)
			}
		}
	}
	return out
}

func isLinkTarget(targets []os.FileInfo, fi os.FileInfo) bool {
	for _, t := range targets {
		if os.SameFile(t, fi) {
			return true
		}
	}
	return false
}

// staleParts lists leftovers whose finished file is already there: partial
// downloads (.part) another run finished first, and failed downloads
// (.unverified) of a file now on disk and checksummed, under any -tag form.
func staleParts(idx *index) []localFile {
	have := map[string]bool{}
	good := map[string]bool{} // top folder + untagged name of checksummed model files
	for _, l := range idx.all {
		have[l.path] = true
		if isModelFile(l.path) && idx.shaByPath[l.path] != "" {
			good[topFolder(idx, l.path)+"/"+untagged(l.path)] = true
		}
	}
	var out []localFile
	for _, l := range idx.all {
		switch {
		case strings.HasSuffix(l.path, ".part") && have[strings.TrimSuffix(l.path, ".part")]:
			out = append(out, l)
		case strings.HasSuffix(l.path, ".unverified"):
			fin := strings.TrimSuffix(l.path, ".unverified")
			if have[fin] || good[topFolder(idx, fin)+"/"+untagged(fin)] {
				out = append(out, l)
			}
		}
	}
	return out
}

// drop takes files out of the index, so the plan works with what stays.
func (idx *index) drop(gone map[string]bool) {
	if len(gone) == 0 {
		return
	}
	keep := func(ls []localFile) []localFile {
		out := ls[:0:0]
		for _, l := range ls {
			if !gone[l.path] {
				out = append(out, l)
			}
		}
		return out
	}
	idx.all = keep(idx.all)
	for k, ls := range idx.byName {
		if idx.byName[k] = keep(ls); len(idx.byName[k]) == 0 {
			delete(idx.byName, k)
		}
	}
	for k, ls := range idx.byHash {
		if idx.byHash[k] = keep(ls); len(idx.byHash[k]) == 0 {
			delete(idx.byHash, k)
		}
	}
	for p := range gone {
		delete(idx.shaByPath, p)
	}
}

// removeCopy deletes one copy of a model file. Its previews and info files
// move to the copy kept when that has none of the same kind, else they go too.
func removeCopy(idx *index, st *stateFile, drop, keep string) (freed int64, err error) {
	oldBase := strings.TrimSuffix(drop, filepath.Ext(drop))
	newBase := strings.TrimSuffix(keep, filepath.Ext(keep))
	for _, sc := range sidecars(drop) {
		dst := newBase + strings.TrimPrefix(sc, oldBase)
		if _, err := os.Lstat(dst); errors.Is(err, os.ErrNotExist) {
			if os.Rename(sc, dst) == nil {
				continue
			}
		}
		if info, err := os.Stat(sc); err == nil {
			freed += info.Size()
		}
		os.Remove(sc)
	}
	info, err := os.Stat(drop)
	if err != nil {
		return freed, err
	}
	if err := os.Remove(drop); err != nil {
		return freed, err
	}
	if e := st.Files[idx.rel(drop)]; e != nil {
		if k := st.Files[idx.rel(keep)]; k != nil && k.Base == "" {
			k.Base, k.BaseName, k.Model, k.VersionID = e.Base, e.BaseName, e.Model, e.VersionID
		}
		delete(st.Files, idx.rel(drop))
	}
	if dir := filepath.Dir(drop); isTagDir(dir) {
		os.Remove(dir) // a tag folder left empty; fails harmlessly otherwise
	}
	return freed + info.Size(), nil
}

// --------------------------------------------------------------- previews

type apiImage struct {
	URL       string          `json:"url"`
	Type      string          `json:"type"`
	NSFW      any             `json:"nsfw"`
	NSFWLevel any             `json:"nsfwLevel"`
	Meta      json.RawMessage `json:"meta"` // how the picture was made, when Civitai knows
	ver       int64           // the version it was listed under
}

// hasMeta reports whether Civitai sent generation data with the image.
func (im apiImage) hasMeta() bool {
	m := strings.TrimSpace(string(im.Meta))
	return m != "" && m != "null" && m != "{}"
}

func (im apiImage) still() bool {
	return im.URL != "" && (im.Type == "" || strings.EqualFold(im.Type, "image"))
}

// sfw reports whether Civitai marks the image safe for work. The level comes
// as a number from some endpoints and as a word from others.
func (im apiImage) sfw() bool {
	switch v := im.NSFWLevel.(type) {
	case float64:
		return v <= 1
	case string:
		return v == "" || strings.EqualFold(v, "None")
	}
	switch v := im.NSFW.(type) {
	case bool:
		return !v
	case string:
		return strings.EqualFold(v, "None") || strings.EqualFold(v, "false")
	}
	return false
}

// previewImages lists the still images for a version, then the rest of its
// base model's versions; safe-for-work images first, Civitai's order kept.
func previewImages(first apiVersion, group []candidate) []apiImage {
	seen := map[string]bool{}
	var out []apiImage
	addFrom := func(v apiVersion) {
		for _, im := range v.Images {
			if im.still() && !seen[im.URL] {
				seen[im.URL] = true
				im.ver = v.ID
				out = append(out, im)
			}
		}
	}
	addFrom(first)
	for _, cd := range group {
		addFrom(cd.ver)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].sfw() && !out[j].sfw() })
	return out
}

// previewStem is where preview number n of a model goes, without extension:
// the first is the one ComfyUI shows (<name>.preview.png), the rest are extras.
func previewStem(modelPath string, n int) string {
	base := strings.TrimSuffix(modelPath, filepath.Ext(modelPath))
	if n == 1 {
		return base + ".preview"
	}
	return fmt.Sprintf("%s.preview-%d", base, n)
}

// previewExists checks for an image at stem.<ext>; for the first preview the
// plain <name>.<ext> that ComfyUI also accepts counts too.
func previewExists(modelPath string, n int) bool {
	stems := []string{previewStem(modelPath, n)}
	if n == 1 {
		stems = append(stems, strings.TrimSuffix(modelPath, filepath.Ext(modelPath)))
	}
	for _, stem := range stems {
		for _, e := range imageExts {
			if _, err := os.Stat(stem + "." + e); err == nil {
				return true
			}
		}
	}
	return false
}

// sidecars lists the files that belong to a model and should follow it when it
// is renamed: previews, numbered previews, and info files from model managers.
func sidecars(modelPath string) []string {
	dir := filepath.Dir(modelPath)
	base := strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasPrefix(name, base+".") || name == filepath.Base(modelPath) {
			continue
		}
		suffix := strings.TrimPrefix(name, base+".")
		ok := suffix == "civitai.info" || suffix == "json" || suffix == "txt"
		for _, x := range append(imageExts, "json") { // pictures and their generation settings
			if suffix == x || suffix == "preview."+x {
				ok = true
			}
			if strings.HasPrefix(suffix, "preview-") && strings.HasSuffix(suffix, "."+x) {
				num := strings.TrimSuffix(strings.TrimPrefix(suffix, "preview-"), "."+x)
				if _, err := strconv.Atoi(num); err == nil {
					ok = true
				}
			}
		}
		if ok {
			out = append(out, filepath.Join(dir, name))
		}
	}
	return out
}

var imageTypes = map[string]string{"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif"}

const maxPreviewBytes = 50 << 20

// fetchImage saves one preview at stem.<ext>, the extension taken from what
// the server says it sent. It never overwrites.
func (c *client) fetchImage(ctx context.Context, rawURL, stem string) (string, error) {
	data, ext, err := c.imageBytes(ctx, rawURL)
	if err != nil {
		return "", err
	}
	dest := stem + "." + ext
	if _, err := os.Lstat(dest); err == nil {
		return dest, nil
	}
	tmp := dest + ".part"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return "", err
	}
	return dest, os.Rename(tmp, dest)
}

// imageBytes downloads one image and says which extension it should have.
func (c *client) imageBytes(ctx context.Context, rawURL string) ([]byte, string, error) {
	u, err := url.Parse(rawURL)
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") {
		return nil, "", fmt.Errorf("unusable image link %q", rawURL)
	}
	var last error
	for attempt := 1; attempt <= 3; attempt++ {
		data, ext, err := func() ([]byte, string, error) {
			rctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
			defer cancel()
			resp, err := c.do(rctx, rawURL, map[string]string{"Accept": "image/*"})
			if err != nil {
				return nil, "", err
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				return nil, "", statusError(resp)
			}
			ct := strings.ToLower(strings.TrimSpace(strings.SplitN(resp.Header.Get("Content-Type"), ";", 2)[0]))
			ext, ok := imageTypes[ct]
			if !ok {
				return nil, "", &httpError{status: http.StatusUnsupportedMediaType}
			}
			data, err := io.ReadAll(io.LimitReader(resp.Body, maxPreviewBytes+1))
			if err != nil {
				return nil, "", err
			}
			if len(data) > maxPreviewBytes {
				return nil, "", fmt.Errorf("image larger than %s", human(maxPreviewBytes))
			}
			return data, ext, nil
		}()
		if err == nil {
			return data, ext, nil
		}
		last = err
		var he *httpError
		if ctx.Err() != nil || (errors.As(err, &he) && !he.retryable()) {
			break
		}
		if attempt < 3 {
			if err := sleepCtx(ctx, backoff(attempt, err)); err != nil {
				return nil, "", err
			}
		}
	}
	var he *httpError
	if errors.As(last, &he) && he.status == http.StatusUnsupportedMediaType {
		return nil, "", errors.New("not an image")
	}
	return nil, "", last
}

// ------------------------------------------------- preview generation settings

// infoPath is where the generation settings of a preview picture go:
// x.preview.jpg -> x.preview.json, x.preview-2.png -> x.preview-2.json.
func infoPath(picture string) string {
	return strings.TrimSuffix(picture, filepath.Ext(picture)) + ".json"
}

// previewPictures lists the preview pictures beside a model that can carry a
// settings file: <name>.preview.<ext> and <name>.preview-N.<ext>. (A plain
// <name>.<ext> cannot: <name>.json is the model's own info file.)
func previewPictures(modelPath string) []string {
	var out []string
	base := strings.TrimSuffix(modelPath, filepath.Ext(modelPath))
	for _, sc := range sidecars(modelPath) {
		suffix := strings.TrimPrefix(sc, base+".")
		ext := strings.TrimPrefix(filepath.Ext(suffix), ".")
		if strings.HasPrefix(suffix, "preview") && slices.Contains(imageExts, strings.ToLower(ext)) {
			out = append(out, sc)
		}
	}
	return out
}

var uuidPart = regexp.MustCompile(`(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

// imageKey names a Civitai image independently of the size and format asked
// for in its link: the id segment of image.civitai.com/<account>/<id>/<options>/<name>.
func imageKey(rawURL string) string {
	u, err := url.Parse(rawURL)
	if err != nil {
		return rawURL
	}
	parts := strings.Split(strings.Trim(u.Path, "/"), "/")
	for _, p := range parts {
		if uuidPart.MatchString(p) {
			return strings.ToLower(p)
		}
	}
	return strings.ToLower(strings.TrimSuffix(parts[len(parts)-1], filepath.Ext(parts[len(parts)-1])))
}

// imageMeta returns the generation data of an image: what came with the model
// listing, or else what Civitai's image gallery has for it (looked up once per
// version, the creator's own posts first).
func (c *client) imageMeta(ctx context.Context, im apiImage, creator string) json.RawMessage {
	if im.hasMeta() {
		return im.Meta
	}
	if im.ver == 0 {
		return nil
	}
	if c.metas == nil {
		c.metas = map[int64]map[string]json.RawMessage{}
	}
	found, ok := c.metas[im.ver]
	if !ok {
		found = map[string]json.RawMessage{}
		users := []string{creator}
		if creator == "" {
			users = []string{""}
		}
		for _, user := range users {
			q := url.Values{"modelVersionId": {strconv.FormatInt(im.ver, 10)}, "limit": {"200"}}
			if user != "" {
				q.Set("username", user)
			}
			for page := 0; page < 3; page++ {
				var resp struct {
					Items    []apiImage `json:"items"`
					Metadata struct {
						NextCursor json.RawMessage `json:"nextCursor"`
					} `json:"metadata"`
				}
				if err := c.getJSON(ctx, c.base.String()+"/api/v1/images?"+q.Encode(), &resp); err != nil {
					break
				}
				for _, it := range resp.Items {
					if it.hasMeta() {
						found[imageKey(it.URL)] = it.Meta
					}
				}
				cursor := strings.Trim(strings.TrimSpace(string(resp.Metadata.NextCursor)), `"`)
				if cursor == "" || cursor == "null" || len(resp.Items) == 0 {
					break
				}
				q.Set("cursor", cursor)
			}
		}
		c.metas[im.ver] = found
	}
	return found[imageKey(im.URL)]
}

// writeImageInfo saves the generation data of a preview beside it. It never
// overwrites a settings file that is there already.
func writeImageInfo(picture, imageURL string, meta json.RawMessage) error {
	dest := infoPath(picture)
	if _, err := os.Lstat(dest); err == nil {
		return nil
	}
	data, err := json.MarshalIndent(struct {
		Source string          `json:"source"`
		URL    string          `json:"url"`
		Meta   json.RawMessage `json:"meta"`
	}{"civitai", imageURL, meta}, "", "  ")
	if err != nil {
		return err
	}
	tmp := dest + ".part"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, dest)
}

// -------------------------------------------------------------- base models

// The base models fetched when -base is not given.
// Common tags: known as tag folder names from the start (see knownTags).
const defaultBases = "flux1_dev,sdxl1,qwen,sd1,krea2"

// Civitai base model name (lower case) -> short tag. A name not listed here
// gets a tag built from the name itself; see baseTag.
var baseAliases = map[string]string{
	"flux.1 d":     "flux1_dev",
	"flux.1 s":     "flux1_schnell",
	"flux.2 d":     "flux2_dev",
	"sdxl 1.0":     "sdxl1",
	"sdxl 1.0 lcm": "sdxl1_lcm",
	"sd 1.4":       "sd1",
	"sd 1.5":       "sd1",
	"sd 1.5 lcm":   "sd1_lcm",
	"sd 1.5 hyper": "sd1_hyper",
}

// Model types that work with any base model: never filtered, never tagged.
var baseAgnostic = map[string]bool{"Upscaler": true}

func hasDigit(s string) bool { return strings.ContainsAny(s, "0123456789") }

// baseTag turns a Civitai base model name into a short tag that is safe in a
// file name: "Flux.1 D" -> flux1_dev, "Krea 2" -> krea2, "Pony V7" -> pony_v7.
// A tag fed back in comes out unchanged, so -base accepts either form.
func baseTag(name string) string {
	s := strings.ToLower(strings.TrimSpace(name))
	if a, ok := baseAliases[s]; ok {
		return a
	}
	var words []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			words = append(words, cur.String())
			cur.Reset()
		}
	}
	for _, r := range s {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'):
			cur.WriteRune(r)
		case r == '.': // "2.1" -> "21"
		default:
			flush()
		}
	}
	flush()
	var parts []string
	for _, w := range words {
		// A bare number joins the word before it ("krea 2" -> "krea2").
		if n := len(parts); n > 0 && !hasDigit(parts[n-1]) && strings.Trim(w, "0123456789") == "" {
			parts[n-1] += w
			continue
		}
		parts = append(parts, w)
	}
	if len(parts) == 0 {
		return "unknown"
	}
	return strings.Join(parts, "_")
}

// Where the base model tag goes when a file is saved or renamed (-tag).
// ----------------------------------------------------------- saved file names

// nameMode says what a downloaded model file is called (-name):
//
//	civitai  the file name Civitai lists, untouched
//	model    built from the model's name and version on Civitai
//	auto     Civitai's name, unless it says nothing (ComfyUI1, lora, a UUID...)
//	         or another liked model uses the same one: then the built name
var nameModes = []string{"auto", "civitai", "model"}
var nameMode = "auto"

var genericStems = map[string]bool{"comfyui": true, "lora": true, "loras": true, "model": true, "models": true,
	"pytorch_lora_weights": true, "lora_weights": true, "adapter_model": true, "diffusion_pytorch_model": true,
	"last": true, "final": true, "output": true, "untitled": true, "checkpoint": true, "test": true, "new": true,
	"my_lora": true, "mylora": true, "unnamed": true, "download": true, "file": true, "style": true, "v": true}

var hexRun = regexp.MustCompile(`(?i)[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}`)

// baseWords are base model names as they turn up in file names; a name made
// of nothing else (sdxl, flux1, pony_v2) says which base, not which model.
var baseWords = map[string]bool{"sd": true, "sd1": true, "sd15": true, "sd21": true, "sd2": true, "sd3": true,
	"sd35": true, "sdxl": true, "xl": true, "flux": true, "flux1": true, "flux2": true, "fluxdev": true, "dev": true,
	"schnell": true, "pony": true, "illustrious": true, "illu": true, "ill": true, "il": true, "noob": true,
	"noobai": true, "zit": true, "zib": true, "zimage": true, "turbo": true, "qwen": true, "krea": true, "chroma": true,
	"wan": true, "ltx": true, "hidream": true, "lora": true, "locon": true, "lycoris": true, "dora": true}

var versionLike = regexp.MustCompile(`^(v|e|ep|epoch|step|steps|r|rank|s)?\d+([._]\d+)*[a-z]?$`)

// genericName reports whether a file name tells nothing about the model: what
// a trainer or ComfyUI saves by default, a UUID, a number, a base model's name,
// or what is left once counters, versions and base model words are taken off
// is five letters or fewer (tst_01, kids4, sdxl, last).
func genericName(name string) bool {
	stem := strings.ToLower(strings.TrimSuffix(name, filepath.Ext(name)))
	if hexRun.MatchString(stem) || strings.HasPrefix(stem, "comfyui") {
		return true
	}
	core := strings.TrimRight(stem, "0123456789_-. ")
	if genericStems[strings.ReplaceAll(core, " ", "_")] {
		return true
	}
	weight := 0
	for _, tok := range strings.FieldsFunc(stem, func(r rune) bool { return r == '_' || r == '-' || r == '.' || r == ' ' }) {
		if baseWords[tok] || knownTags[tok] || versionLike.MatchString(tok) {
			continue
		}
		for _, r := range tok {
			switch {
			case r < 128 && unicode.IsLetter(r):
				weight++
			case unicode.IsLetter(r):
				weight += 3 // one CJK character says about as much as a short word
			}
		}
	}
	return weight <= 5
}

// ------------------------------------------------------------- display titles

// titleBases are the base model names authors tack onto model titles:
// "Crystals [FLUX+SDXL+PONY+1.5]", "Style XL + F1D + Illu + Pony + zit + zib".
var titleBases = map[string]bool{"flux": true, "fl": true, "f1d": true, "f1": true, "f.1": true, "flux1": true,
	"flux.1": true, "flux.1d": true, "fluxdev": true, "dev": true, "schnell": true, "klein": true, "kontext": true,
	"9b": true, "4b": true, "krea": true, "krea2": true, "sdxl": true, "xl": true, "sd": true, "sd1": true,
	"sd1.5": true, "sd15": true, "1.5": true, "sd2": true, "sd2.1": true, "sd3": true, "sd3.5": true, "sd3.5l": true,
	"sd35": true, "pony": true, "pd": true, "ponyxl": true, "il": true, "ill": true, "illu": true, "illustrious": true,
	"noob": true, "noobai": true, "zit": true, "zib": true, "zimage": true, "z-image": true, "turbo": true,
	"qwen": true, "chroma": true, "anima": true, "wan": true, "hunyuan": true, "ltx": true, "hidream": true,
	"lumina": true, "auraflow": true, "base": true, "and": true}

// titleFillers may sit in such a list without making it anything else.
var titleFillers = map[string]bool{"lora": true, "loha": true, "locon": true, "lycoris": true, "dora": true,
	"for": true, "z": true, "image": true}

var titleVersion = regexp.MustCompile(`(?i)^v?\d+(\.\d+)*[a-z]?$`)

// allBases reports whether text is nothing but base model names, versions and
// joiners, and how many base names it holds.
func allBases(text string) (bool, int) {
	n := 0
	for _, tok := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return unicode.IsSpace(r) || strings.ContainsRune("+,/&|_-", r)
	}) {
		tok = strings.Trim(tok, ".")
		switch {
		case tok == "" || titleFillers[tok]:
		case titleBases[tok]:
			n++
		case titleVersion.MatchString(tok):
		default:
			return false, n
		}
	}
	return true, n
}

// cleanTitle is a model's name without the base model list its author put at
// the end: "1950's (Technicolor) style XL + F1D + Illu + Pony + zit + zib"
// becomes "1950's (Technicolor) style". A single base word that may belong to
// the name ("Juggernaut XL") stays; a bracketed or joined list goes.
func cleanTitle(name string) string {
	trimEnd := func(s string) string {
		return strings.TrimRightFunc(s, func(r rune) bool {
			return !(unicode.IsLetter(r) || unicode.IsDigit(r) || r == ')' || r == ']' || r == '\'' || r == '!' || r == '?')
		})
	}
	s := trimEnd(strings.TrimSpace(name))
	for {
		before := s
		// A bracketed list at the end: [FLUX+SDXL+PONY+1.5], (SDXL/Pony), (+SDXL).
		if strings.HasSuffix(s, "]") || strings.HasSuffix(s, ")") {
			open := strings.LastIndexAny(s, "[(")
			if open > 0 {
				if ok, n := allBases(s[open+1 : len(s)-1]); ok && n > 0 {
					s = trimEnd(s[:open])
				}
			}
		}
		// A list after a separator: "Painted World | Pony, SDXL, Flux".
		for _, sep := range []string{" | ", "|", " - ", " – ", " — ", " / "} {
			if i := strings.LastIndex(s, sep); i > 0 {
				if ok, n := allBases(s[i+len(sep):]); ok && n > 0 {
					s = trimEnd(s[:i])
					break
				}
			}
		}
		// A run of base names at the end joined by + , / or two or more of them:
		// "Style XL + F1D + Illu + Pony + zit". One bare "XL" stays.
		words := strings.Fields(s)
		cut := len(words)
		for cut > 0 {
			if ok, _ := allBases(words[cut-1]); !ok {
				break
			}
			cut--
		}
		for cut < len(words) && titleVersion.MatchString(words[cut]) && !titleBases[strings.ToLower(words[cut])] {
			cut++ // a number right after the name is part of it: "Paradox 3 SD XL 1.0"
		}
		if cut > 0 && cut < len(words) {
			tail := strings.Join(words[cut:], " ")
			_, n := allBases(tail)
			if n >= 2 || strings.ContainsAny(tail, "+,/") {
				s = trimEnd(strings.Join(words[:cut], " "))
			}
		}
		if s == before || s == "" {
			break
		}
	}
	if s == "" {
		return strings.TrimSpace(name)
	}
	return s
}

// cleanName turns a model name and version into a file name: letters (any
// script), digits, - and . stay; spaces and the rest become single underscores;
// brackets, quotes, slashes and emoji go.
func cleanName(parts ...string) string {
	var b strings.Builder
	under := false
	for _, p := range parts {
		for _, r := range p {
			switch {
			case r == '\'' || r == '’' || r == '`':
				// Zavy's -> Zavys
			case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' || r == '.':
				b.WriteRune(r)
				under = false
			case !under && b.Len() > 0:
				b.WriteByte('_')
				under = true
			}
		}
		if !under && b.Len() > 0 {
			b.WriteByte('_')
			under = true
		}
	}
	s := strings.Trim(b.String(), "_-.")
	for strings.Contains(s, "_-_") {
		s = strings.ReplaceAll(s, "_-_", "-")
	}
	if r := []rune(s); len(r) > 120 {
		s = strings.TrimRight(string(r[:120]), "_-.")
	}
	return s
}

// sharedNames counts, per lower-case Civitai file name, the liked models using it.
func sharedNames(models []apiModel) map[string]int {
	ids := map[string]map[int64]bool{}
	for _, m := range models {
		for _, v := range m.ModelVersions {
			if f, ok := modelFile(v); ok {
				k := strings.ToLower(f.Name)
				if ids[k] == nil {
					ids[k] = map[int64]bool{}
				}
				ids[k][m.ID] = true
			}
		}
	}
	out := map[string]int{}
	for k, s := range ids {
		out[k] = len(s)
	}
	return out
}

// saveName is the name a model file is saved under, and why it is not
// Civitai's own (empty when it is).
func saveName(m apiModel, v apiVersion, f apiFile, shared map[string]int) (string, string) {
	why := ""
	switch nameMode {
	case "civitai":
		return f.Name, ""
	case "model":
		why = "-name model"
	default:
		switch {
		case genericName(f.Name):
			why = fmt.Sprintf("Civitai's file name %q says nothing about it", f.Name)
		case shared[strings.ToLower(f.Name)] > 1:
			why = fmt.Sprintf("Civitai's file name %q is used by %d liked models", f.Name, shared[strings.ToLower(f.Name)])
		default:
			return f.Name, ""
		}
	}
	built := modelFileName(m.Name, v.Name)
	if built == "" {
		return f.Name, ""
	}
	return built + strings.ToLower(filepath.Ext(f.Name)), why
}

// modelFileName builds a file name (no extension) from a model's name and
// version, leaving the version out when the model name already says it.
func modelFileName(model, version string) string {
	model = cleanTitle(model)
	if v := strings.ToLower(strings.TrimSpace(version)); v == "" || strings.Contains(strings.ToLower(model), v) {
		return cleanName(model)
	}
	return cleanName(model, version)
}

// localSaveName is the name for a model file that is not a liked model but
// that Civitai knows by its checksum, under -name (current: its name now,
// without a base model tag).
func localSaveName(current, model, version string) (string, string) {
	if model == "" || nameMode == "civitai" {
		return current, ""
	}
	why := "-name model"
	if nameMode == "auto" {
		if !genericName(current) {
			return current, ""
		}
		why = fmt.Sprintf("its file name %q says nothing about it", current)
	}
	built := modelFileName(model, version)
	if built == "" {
		return current, ""
	}
	return built + strings.ToLower(filepath.Ext(current)), why
}

var tagModes = []string{"front", "end", "dot", "folder", "none"}
var tagMode = "front"

func noTag(tag string) bool { return tag == "" || tag == "unknown" || tag == "other" }

// tagForms lists every name a model file may have on disk: Civitai's own
// name and each tagged form, so a file is recognised whichever -tag it was
// saved with. Already-tagged Civitai names are not tagged twice.
func tagForms(tag, name string) []string {
	forms := []string{name}
	if noTag(tag) {
		return forms
	}
	stem, ext := strings.TrimSuffix(name, filepath.Ext(name)), filepath.Ext(name)
	low := strings.ToLower(stem)
	if !strings.HasPrefix(low, tag+"_") {
		forms = append(forms, tag+"_"+name)
	}
	if !strings.HasSuffix(low, "_"+tag) {
		forms = append(forms, stem+"_"+tag+ext)
	}
	if !strings.HasSuffix(low, "."+tag) {
		forms = append(forms, stem+"."+tag+ext)
	}
	return forms
}

// taggedName is the file name a model file gets on disk under -tag; for
// "folder" the tag is a subfolder instead (see tagSubdir).
func taggedName(tag, name string) string {
	if noTag(tag) {
		return name
	}
	stem, ext := strings.TrimSuffix(name, filepath.Ext(name)), filepath.Ext(name)
	low := strings.ToLower(stem)
	switch tagMode {
	case "front":
		if strings.HasPrefix(low, tag+"_") {
			return name
		}
		return tag + "_" + name
	case "end":
		if strings.HasSuffix(low, "_"+tag) {
			return name
		}
		return stem + "_" + tag + ext
	case "dot":
		if strings.HasSuffix(low, "."+tag) {
			return name
		}
		return stem + "." + tag + ext
	}
	return name // folder, none
}

func tagSubdir(tag string) string {
	if tagMode == "folder" && !noTag(tag) {
		return tag
	}
	return ""
}

// knownTags holds every base model tag the tool could have used in a name or
// as a folder: the defaults, the aliases, -base, and the bases of the liked
// models (filled in by run). A folder with one of these names is a tag folder.
var knownTags = map[string]bool{}

func init() {
	for _, t := range strings.Split(defaultBases, ",") {
		knownTags[t] = true
	}
	for _, t := range baseAliases {
		knownTags[t] = true
	}
}

func isTagDir(dir string) bool { return knownTags[strings.ToLower(filepath.Base(dir))] }

// desiredPath is where a file already on disk should be under -tag, staying
// in the folder it was filed in. A tag folder it sits in is replaced, never
// nested: loras/sdxl1/x for a Pony file becomes loras/pony/x, not
// loras/sdxl1/pony/x; outside folder mode it backs out of the tag folder.
func desiredPath(current, tag, civitaiName string) string {
	dir := filepath.Dir(current)
	if isTagDir(dir) || (!noTag(tag) && strings.EqualFold(filepath.Base(dir), tag)) {
		if tagMode == "folder" && strings.EqualFold(filepath.Base(dir), tag) {
			return filepath.Join(dir, taggedName(tag, civitaiName)) // already in its own folder
		}
		dir = filepath.Dir(dir)
	}
	return filepath.Join(dir, tagSubdir(tag), taggedName(tag, civitaiName))
}

// parseBases reads the -base value: a comma list of tags or Civitai names
// ("+tag" is the same as "tag"). Empty or "all" selects every base model.
func parseBases(spec string) (set map[string]bool, all bool) {
	set = map[string]bool{}
	for _, p := range strings.Split(spec, ",") {
		p = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(p), "+"))
		switch {
		case p == "":
		case strings.EqualFold(p, "all"):
			all = true
		default:
			set[baseTag(p)] = true
		}
	}
	if len(set) == 0 {
		all = true
	}
	return set, all
}

// ------------------------------------------------------------ early access

// eaRecord remembers a version whose download was refused while it was in
// early access, so it is not tried again before the early access ends.
type eaRecord struct {
	Model     string    `json:"model"`
	Version   string    `json:"version"`
	Tag       string    `json:"tag,omitempty"`
	Reason    string    `json:"reason"`
	BlockedAt time.Time `json:"blockedAt"`
	EndsAt    string    `json:"endsAt,omitempty"` // as Civitai gave it; "" = Civitai gave no end date
	// What Civitai said about the version when it was refused, for diagnosis.
	Availability string `json:"availability,omitempty"`
	UsageControl string `json:"usageControl,omitempty"`
}

func parseTime(s string) (time.Time, bool) {
	if s = strings.TrimSpace(s); s == "" {
		return time.Time{}, false
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05.000Z", "2006-01-02"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t, true
		}
	}
	return time.Time{}, false
}

// earlyAccess reports whether Civitai currently has the version in early access.
func (v apiVersion) earlyAccess(now time.Time) bool {
	if strings.EqualFold(v.Availability, "EarlyAccess") {
		return true
	}
	if t, ok := parseTime(v.EarlyAccessEndsAt); ok {
		return now.Before(t)
	}
	return false
}

func eaUntil(endsAt string) string {
	if t, ok := parseTime(endsAt); ok {
		return t.Local().Format("Jan 2 2006 15:04")
	}
	return "an unknown date"
}

// blockedByAccess tells a refusal (not paid for, not allowed) apart from a
// missing file or a network problem.
func blockedByAccess(err error) bool {
	var he *httpError
	if errors.As(err, &he) {
		return he.status == http.StatusUnauthorized || he.status == http.StatusPaymentRequired ||
			he.status == http.StatusForbidden
	}
	return errors.Is(err, errWall)
}

// versionDetails asks Civitai for one version's current early access state.
func (c *client) versionDetails(ctx context.Context, id int64) (apiVersion, error) {
	var v apiVersion
	err := c.getJSON(ctx, fmt.Sprintf("%s/api/v1/model-versions/%d", c.base.String(), id), &v)
	return v, err
}

// applyEarlyAccess looks up the early access state of everything about to be
// downloaded and holds back versions refused before, until their early access
// ends or -ea asks for another try.
func applyEarlyAccess(ctx context.Context, c *client, st *stateFile, items []item, retry bool, now time.Time) {
	for i := range items {
		it := &items[i]
		if it.status != stGet {
			continue
		}
		if d, err := c.versionDetails(ctx, it.ver.ID); err == nil {
			if d.Availability != "" {
				it.ver.Availability = d.Availability
			}
			if d.EarlyAccessEndsAt != "" {
				it.ver.EarlyAccessEndsAt = d.EarlyAccessEndsAt
			}
			if d.UsageControl != "" {
				it.ver.UsageControl = d.UsageControl
			}
		}
		if ctx.Err() != nil {
			return
		}
		it.ea = it.ver.earlyAccess(now)
		rec := st.EarlyAccess[strconv.FormatInt(it.ver.ID, 10)]
		if rec == nil {
			continue
		}
		it.eaBlocked = true
		ends := it.ver.EarlyAccessEndsAt
		if ends == "" {
			ends = rec.EndsAt
		}
		t, known := parseTime(ends)
		refused := fmt.Sprintf("refused %s (%s)", rec.BlockedAt.Local().Format("Jan 2"), rec.Reason)
		switch {
		case retry:
			it.note = joinNote(it.note, "retry; "+refused)
		case it.ea || (known && now.Before(t)):
			it.status = stWait
			it.note = fmt.Sprintf("%s; early access ends %s", refused, eaUntil(ends))
		case !known:
			// Civitai gives no end date, so there is nothing to wait for:
			// held until -ea asks for another try.
			it.status = stWait
			it.note = refused + "; Civitai gives no end date, held until -ea"
		default:
			it.note = joinNote(it.note, "early access has ended, trying again")
		}
	}
}

func joinNote(a, b string) string {
	if a == "" {
		return b
	}
	return a + "; " + b
}

// -------------------------------------------------------------------- plan

const (
	stHave  = "have"  // already on disk
	stAlone = "alone" // something is in the way; left untouched
	stSkip  = "skip"  // nothing to download
	stGet   = "get"   // will be downloaded
	stWait  = "wait"  // early access refused before; waiting for it to end
	stOff   = "off"   // its base model is not downloaded (listed in the config only)
)

type thumbJob struct {
	url  string
	stem string // path without extension
	slot int
	img  apiImage
}

type item struct {
	model     apiModel
	ver       apiVersion
	file      apiFile
	tag       string // base model tag, "" for types without one
	status    string
	note      string
	dest      string      // where a download goes (stGet)
	found     string      // where the file already is (stHave)
	rename    string      // the name `found` should have to carry its tag; "" if it does
	images    []apiImage  // preview candidates, best first
	thumbs    []thumbJob  // previews to fetch
	sidecars  [][2]string // files that move along with a rename
	noInfo    []string    // preview pictures on disk with no settings file (-preview-info)
	ea        bool        // Civitai has this version in early access right now
	eaBlocked bool        // a download of it was refused before (recorded in the state file)
	local     bool        // not a liked model: found on Civitai by its checksum (-local)
}

// final is where the model file is, or will be, once this run is done.
func (it item) final(renaming bool) string {
	switch {
	case it.status == stGet:
		return it.dest
	case renaming && it.rename != "":
		return it.rename
	}
	return it.found
}

type candidate struct {
	ver    apiVersion
	file   apiFile
	tag    string
	tagged string // file name on disk: base tag + the name it is saved under
	save   string // the name it is saved under (-name); file.Name is Civitai's
	why    string // why save differs from Civitai's name
}

type planOptions struct {
	latest   bool
	bases    map[string]bool
	all      bool
	rename   bool
	previews int
	unsure   map[string]bool // files that fit several liked files by name and size, with no checksum
	// From the config file:
	blocked      map[string]bool // modelKey -> "download": false for that base
	basePreviews map[string]int  // base tag -> previews for it
	baseLatest   map[string]bool // base tag -> latest for it
	removing     map[string]bool // copies and leftovers this run removes first
	links        []os.FileInfo   // files symbolic links point to: never renamed
}

// previewsFor is how many previews a file of this base model keeps.
func (o planOptions) previewsFor(tag string) int {
	if n, ok := o.basePreviews[tag]; ok {
		return n
	}
	return o.previews
}

type planResult struct {
	items   []item
	others  map[string]int    // base tag not selected -> liked models that have it
	names   map[string]string // base tag -> what Civitai calls it
	matched map[string]bool   // selected tags that at least one liked model has
	off     []item            // liked models of base models not handled (for the config's lists)
}

// names lists every file name a copy of this candidate may have on disk: its
// saved name and Civitai's, each in every -tag form.
func (cd candidate) names() []string {
	out := tagForms(cd.tag, cd.file.Name)
	if cd.save != cd.file.Name {
		out = append(out, tagForms(cd.tag, cd.save)...)
	}
	return out
}

// modelFile picks the file of a version that is the model itself.
func modelFile(v apiVersion) (apiFile, bool) {
	pick := -1
	for i, f := range v.Files {
		if f.Primary {
			pick = i
			break
		}
	}
	if pick < 0 {
		for i, f := range v.Files {
			if f.Type == "Model" || f.Type == "Pruned Model" {
				pick = i
				break
			}
		}
	}
	if pick < 0 {
		return apiFile{}, false
	}
	f := v.Files[pick]
	// Never trust a remote name as a path.
	f.Name = filepath.Base(strings.ReplaceAll(strings.TrimSpace(f.Name), `\`, "/"))
	if f.Name == "" || f.Name == "." || f.Name == ".." || f.Name == "/" || f.DownloadURL == "" {
		return apiFile{}, false
	}
	return f, true
}

func versionLabel(v apiVersion) string {
	if v.Name == "" {
		return ""
	}
	return " (" + v.Name + ")"
}

func plan(c *client, idx *index, st *stateFile, models []apiModel, opt planOptions) *planResult {
	res := &planResult{others: map[string]int{}, names: map[string]string{}, matched: map[string]bool{}}
	planned := map[string]string{} // lower-case file name -> model that claimed it
	shared := sharedNames(models)

	// evaluate decides what to do about one model for one base model.
	// cands are that base's versions, newest first.
	evaluate := func(m apiModel, dir string, cands []candidate) item {
		target := cands[0]
		if dir == "checkpoints" && strings.EqualFold(filepath.Ext(target.file.Name), ".gguf") {
			dir = "diffusion_models" // GGUF models load through Unet Loader (GGUF), not as checkpoints
		}
		it := item{model: m, ver: target.ver, file: target.file, tag: target.tag}
		it.images = previewImages(target.ver, cands)
		check := cands
		if opt.latest || opt.baseLatest[target.tag] {
			check = cands[:1]
		}
		have := func(l localFile, cd candidate, how string, i int) item {
			it.status, it.found = stHave, l.path
			it.images = previewImages(cd.ver, cands)
			it.note = idx.rel(l.path) + how
			if i > 0 {
				it.note += fmt.Sprintf("; that is version %q, the newest is %q", cd.ver.Name, target.ver.Name)
			}
			if want := ggufHome(idx, desiredPath(l.path, cd.tag, cd.save)); !strings.EqualFold(want, l.path) {
				it.rename = want
				if cd.why != "" && !strings.EqualFold(untagged(want), untagged(l.path)) {
					it.note = joinNote(it.note, "new name from the model: "+cd.why)
				}
			}
			return it
		}

		// 1. On disk under its tagged name or under Civitai's own name, anywhere
		//    in the tree, with the size agreeing.
		//    Where both checksums are known they must agree: an SDXL and a Pony
		//    version of a LoRA can share a name and a size.
		for i, cd := range check {
			for _, name := range cd.names() {
				for _, l := range idx.byName[strings.ToLower(name)] {
					if cd.file.bytes() != 0 && !sizeClose(l.size, cd.file.bytes()) {
						continue
					}
					if want, got := cd.file.sha256(), idx.shaByPath[l.path]; want != "" && got != "" && want != got {
						continue
					}
					if opt.unsure[l.path] {
						it.status = stAlone
						it.note = idx.rel(l.path) + " fits more than one liked file by name and size (two base models' versions?); not checksummed this run, so not sure which"
						return it
					}
					return have(l, cd, "", i)
				}
			}
		}
		// 2. The name it would be saved under belongs to a different file.
		//    In folder mode the same name in another base model's folder is fine.
		for _, l := range idx.byName[strings.ToLower(target.tagged)] {
			if tagMode == "folder" && !noTag(target.tag) && !strings.EqualFold(filepath.Base(filepath.Dir(l.path)), target.tag) {
				continue
			}
			it.status = stAlone
			if sizeClose(l.size, target.file.bytes()) {
				it.note = fmt.Sprintf("%s already has the name this would be saved under, but is a different file (checksums differ)", idx.rel(l.path))
			} else {
				it.note = fmt.Sprintf("%s already has the name this would be saved under, with a different size (yours %s, Civitai's %s)",
					idx.rel(l.path), human(l.size), human(target.file.bytes()))
			}
			return it
		}
		// 3. An earlier download of it that failed its checksum: downloaded
		//    again below (unless a good copy turns up), and the bad one goes
		//    once the new one checks out.
		retry := ""
		if ls := idx.byName[strings.ToLower(target.tagged)+".unverified"]; len(ls) > 0 {
			retry = idx.rel(ls[0].path)
		}
		// 4. A copy under some other name, by the checksums on record.
		for i, cd := range check {
			if sha := cd.file.sha256(); sha != "" {
				for _, l := range idx.byHash[sha] {
					return have(l, cd, fmt.Sprintf(" (same file under another name; Civitai calls it %s)", cd.file.Name), i)
				}
			}
		}
		// 5. A model file of exactly the same size with no checksum on record:
		//    probably a copy, so it is not downloaded until that is settled.
		for _, cd := range check {
			want := cd.file.bytes()
			if want == 0 {
				continue
			}
			for _, l := range idx.all {
				if !isModelFile(l.path) || !sizeExact(l.size, want) {
					continue
				}
				if cd.file.sha256() != "" && idx.shaByPath[l.path] != "" {
					continue // both checksums known and step 4 found them different
				}
				it.status = stAlone
				if cd.file.sha256() == "" {
					it.note = fmt.Sprintf("%s is the same size and Civitai gave no checksum to tell them apart", idx.rel(l.path))
				} else {
					it.note = fmt.Sprintf("%s is exactly the same size and not checksummed this run, so it may be this file; not downloaded twice", idx.rel(l.path))
				}
				return it
			}
		}
		// 6. Missing: download the newest version for this base model.
		key := strings.ToLower(filepath.Join(dir, tagSubdir(target.tag), target.tagged))
		u, err := url.Parse(target.file.DownloadURL)
		switch {
		case err != nil || !strings.EqualFold(u.Host, c.base.Host) || u.Scheme != c.base.Scheme:
			it.status, it.note = stSkip, "its download link does not point at Civitai"
		case planned[key] != "":
			it.status = stAlone
			it.note = fmt.Sprintf("%q would be saved under the same name %s; only that one is downloaded", planned[key], target.tagged)
		default:
			planned[key] = m.Name
			it.status = stGet
			it.dest = filepath.Join(idx.root, dir, tagSubdir(target.tag), target.tagged)
			if ls := idx.byName[strings.ToLower(target.file.Name)]; len(ls) > 0 && target.tagged != target.file.Name {
				it.note = fmt.Sprintf("%s has Civitai's name for it but is a different file (another base model's version, or another size)", idx.rel(ls[0].path))
			}
			if retry != "" {
				it.note = joinNote(it.note, retry+" failed its checksum before; downloading it again")
			}
			if target.why != "" {
				it.note = joinNote(it.note, "named from the model: "+target.why)
			}
		}
		return it
	}

	for _, m := range models {
		dir, ok := typeDirs[m.Type]
		if !ok {
			res.items = append(res.items, item{model: m, status: stSkip,
				note: fmt.Sprintf("type %q is not a model file ComfyUI loads", m.Type)})
			continue
		}
		// Group the model's versions by base model, keeping newest-first order.
		agnostic := baseAgnostic[m.Type]
		var order []string
		groups := map[string][]candidate{}
		for _, v := range m.ModelVersions {
			f, ok := modelFile(v)
			if !ok {
				continue
			}
			tag := ""
			if !agnostic {
				tag = baseTag(v.BaseModel)
				if res.names[tag] == "" {
					res.names[tag] = strings.TrimSpace(v.BaseModel)
				}
			}
			if _, seen := groups[tag]; !seen {
				order = append(order, tag)
			}
			save, why := saveName(m, v, f, shared)
			groups[tag] = append(groups[tag], candidate{ver: v, file: f, tag: tag, tagged: taggedName(tag, save), save: save, why: why})
		}
		if len(order) == 0 {
			note := "no downloadable file"
			if m.Mode != "" {
				note += " (" + strings.ToLower(m.Mode) + " on Civitai)"
			}
			res.items = append(res.items, item{model: m, status: stSkip, note: note})
			continue
		}
		if agnostic && !opt.all && !opt.bases[""] {
			res.others[""]++ // no base model (upscalers), not selected
			res.off = append(res.off, item{model: m, ver: groups[""][0].ver, status: stOff})
			continue
		}
		for _, tag := range order {
			if !agnostic && !opt.all && !opt.bases[tag] {
				res.others[tag]++ // not handled at all this run
				res.off = append(res.off, item{model: m, ver: groups[tag][0].ver, tag: tag, status: stOff})
				continue
			}
			res.matched[tag] = true
			it := evaluate(m, dir, groups[tag])
			if it.status == stGet && opt.blocked[modelKey(tag, m.ID)] {
				it.status, it.dest = stSkip, ""
				it.note = "not downloaded: \"download\" is false for it in the config"
			}
			res.items = append(res.items, it)
		}
	}
	settle(res, st, idx, opt, 0)
	return res
}

// settle works out the renames and previews for res.items[from:]; items
// before from have been settled already and keep their claims.
func settle(res *planResult, st *stateFile, idx *index, opt planOptions, from int) {
	// A file is renamed only when nothing has the new name and no other model
	// lays claim to the same file or the same new name. Its previews and info
	// files go with it.
	srcTaken, dstTaken := map[string]bool{}, map[string]bool{}
	for _, it := range res.items[:from] {
		if it.status == stHave && it.rename != "" {
			srcTaken[it.found], dstTaken[strings.ToLower(it.rename)] = true, true
			for _, sc := range it.sidecars {
				dstTaken[strings.ToLower(sc[1])] = true
			}
		}
	}
	for i := from; i < len(res.items); i++ {
		it := &res.items[i]
		if it.status != stHave || it.rename == "" {
			continue
		}
		to := it.rename
		if fi, err := os.Stat(it.found); err == nil && isLinkTarget(opt.links, fi) {
			it.note += "; not renamed, a link points to it"
			it.rename = ""
			continue
		}
		_, err := os.Lstat(to)
		if (err == nil && !opt.removing[to]) || srcTaken[it.found] || dstTaken[strings.ToLower(to)] {
			it.note += fmt.Sprintf("; cannot be renamed, %s is taken", idx.rel(it.rename))
			it.rename = ""
			continue
		}
		srcTaken[it.found], dstTaken[strings.ToLower(to)] = true, true
		oldBase := strings.TrimSuffix(it.found, filepath.Ext(it.found))
		newBase := strings.TrimSuffix(to, filepath.Ext(to))
		for _, sc := range sidecars(it.found) {
			dst := newBase + strings.TrimPrefix(sc, oldBase)
			if _, err := os.Lstat(dst); err == nil || dstTaken[strings.ToLower(dst)] {
				continue
			}
			dstTaken[strings.ToLower(dst)] = true
			it.sidecars = append(it.sidecars, [2]string{sc, dst})
		}
	}

	// Previews: for every file kept or downloaded, fill the preview slots that
	// are empty with images not already saved for it.
	{
		for i := from; i < len(res.items); i++ {
			it := &res.items[i]
			if it.status != stHave && it.status != stGet {
				continue
			}
			keep := opt.previewsFor(it.tag)
			final := it.final(opt.rename)
			current := final // where its previews are right now
			if it.status == stHave {
				current = it.found
			}
			used := map[string]bool{}
			if e := st.Files[idx.rel(current)]; e != nil {
				for _, u := range e.Previews {
					used[u] = true
				}
			}
			next := 0
			for slot := 1; slot <= keep; slot++ {
				if previewExists(current, slot) {
					continue
				}
				for next < len(it.images) && used[it.images[next].URL] {
					next++
				}
				if next >= len(it.images) {
					break
				}
				it.thumbs = append(it.thumbs, thumbJob{url: it.images[next].URL, stem: previewStem(final, slot), slot: slot,
					img: it.images[next]})
				used[it.images[next].URL] = true
				next++
			}
		}
	}
}

// ------------------------------------------------------------- local files

// Folders whose files -local looks up on Civitai. Upscalers have no base
// model; text encoders and the like are not Civitai models and are big.
var localDirs = map[string]bool{"checkpoints": true, "loras": true, "diffusion_models": true, "unet": true,
	"embeddings": true, "controlnet": true, "hypernetworks": true, "vae": true, "animatediff_models": true}

// hashVersion is what Civitai returns for a file looked up by its checksum.
type hashVersion struct {
	apiVersion
	ModelID int64 `json:"modelId"`
	Model   struct {
		Name string `json:"name"`
		Type string `json:"type"`
	} `json:"model"`
}

// versionByHash asks Civitai which model version a file is; nil when Civitai
// does not know the file.
func (c *client) versionByHash(ctx context.Context, sha string) (*hashVersion, error) {
	var v hashVersion
	err := c.getJSON(ctx, c.base.String()+"/api/v1/model-versions/by-hash/"+url.PathEscape(sha), &v)
	var he *httpError
	if errors.As(err, &he) && he.status == http.StatusNotFound {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &v, nil
}

// untag strips a tag the name already carries in any -tag form, so a file
// that was tagged one way can be retagged another.
func untag(tag, name string) string {
	stem, ext := strings.TrimSuffix(name, filepath.Ext(name)), filepath.Ext(name)
	low := strings.ToLower(stem)
	switch {
	case len(stem) > len(tag)+1 && strings.HasPrefix(low, tag+"_"):
		stem = stem[len(tag)+1:]
	case len(stem) > len(tag)+1 && (strings.HasSuffix(low, "_"+tag) || strings.HasSuffix(low, "."+tag)):
		stem = stem[:len(stem)-len(tag)-1]
	}
	return stem + ext
}

type leftFile struct{ rel, why string }

type localStats struct {
	found, guessed, failed int
	left                   []leftFile // model files still without a known base, and why
	notLooked              int        // -local off: files no liked model accounts for, base unknown
}

// guessBase reads the base model from a safetensors file's own metadata, as
// trainers write it (ss_base_model_version, modelspec.architecture). Only what
// the metadata states is used, never a guess from the tensor layout.
func guessBase(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	if ext != ".safetensors" && ext != ".sft" {
		return ""
	}
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	var n uint64
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil || n == 0 || n > 64<<20 {
		return ""
	}
	var header struct {
		Meta map[string]any `json:"__metadata__"`
	}
	if err := json.NewDecoder(io.LimitReader(f, int64(n))).Decode(&header); err != nil {
		return ""
	}
	str := func(k string) string {
		v, _ := header.Meta[k].(string)
		return strings.ToLower(v)
	}
	for _, text := range []string{str("modelspec.architecture"), str("ss_base_model_version")} {
		switch {
		case text == "":
		case strings.Contains(text, "krea") && strings.Contains(text, "2"):
			return "krea2"
		case strings.Contains(text, "schnell"):
			return "flux1_schnell"
		case strings.Contains(text, "flux-2") || strings.Contains(text, "flux2"):
			return "flux2_dev"
		case strings.Contains(text, "flux"):
			return "flux1_dev"
		case strings.Contains(text, "xl"):
			return "sdxl1"
		case strings.Contains(text, "qwen"):
			return "qwen"
		case strings.Contains(text, "v1"):
			return "sd1"
		}
	}
	return ""
}

// ggufHome moves a .gguf diffusion model out of checkpoints: ComfyUI loads
// GGUF models from diffusion_models (Unet Loader (GGUF)), not as checkpoints.
func ggufHome(idx *index, path string) string {
	if !strings.EqualFold(filepath.Ext(path), ".gguf") {
		return path
	}
	rel := filepath.ToSlash(idx.rel(path))
	if top, rest, ok := strings.Cut(rel, "/"); ok && top == "checkpoints" {
		return filepath.Join(idx.root, "diffusion_models", filepath.FromSlash(rest))
	}
	return path
}

// carriesTag says whether a file name already has one of the tags in it, in
// any -tag form, or sits in a tag folder.
func carriesTag(path string, tags map[string]bool) bool {
	stem := strings.ToLower(strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)))
	dir := strings.ToLower(filepath.Base(filepath.Dir(path)))
	for t := range tags {
		if noTag(t) {
			continue
		}
		if dir == t || strings.HasPrefix(stem, t+"_") || strings.HasSuffix(stem, "_"+t) || strings.HasSuffix(stem, "."+t) {
			return true
		}
	}
	return false
}

// lookupLocal deals with model files that no liked model accounts for. With
// -local it asks Civitai what they are by checksum, so they can be tagged and
// previewed like the rest; with -guess, files Civitai does not know are tagged
// from their own metadata. Checksums and answers are kept in the state file:
// a file is checksummed once, and one Civitai does not know is not asked about
// again until it changes. Whatever is left without a base is listed, with why.
func lookupLocal(ctx context.Context, c *client, idx *index, st *stateFile, res *planResult, opt planOptions, enabled, guess bool) (localStats, error) {
	var ls localStats
	claimed := map[string]bool{}
	for _, it := range res.items {
		if it.found != "" {
			claimed[it.found] = true
		}
	}
	known := map[string]bool{} // every tag the tool could have put in a name
	for t := range opt.bases {
		known[t] = true
	}
	for t := range res.names {
		known[t] = true
	}
	for t := range knownTags {
		known[t] = true
	}
	var todo []localFile
	for _, l := range idx.all {
		rel := idx.rel(l.path)
		top, _, _ := strings.Cut(filepath.ToSlash(rel), "/")
		if !isModelFile(l.path) || claimed[l.path] || !localDirs[top] || st.Files[rel] == nil {
			continue
		}
		todo = append(todo, l)
	}

	if enabled {
		// Checksum what has none yet.
		var hashN int
		var hashBytes int64
		need := func(e *fileState) bool { return e.SHA256 == "" && !e.NotOnCivitai && e.VersionID == 0 }
		for _, l := range todo {
			if need(st.Files[idx.rel(l.path)]) {
				hashN++
				hashBytes += l.size
			}
		}
		if hashN > 0 {
			fmt.Fprintf(os.Stderr, "Checksumming %d model file(s) no liked model accounts for, %s, to look them up on Civitai...\n", hashN, human(hashBytes))
		}
		done := 0
		for _, l := range todo {
			e := st.Files[idx.rel(l.path)]
			if !need(e) {
				continue
			}
			done++
			fmt.Fprintf(os.Stderr, "  [%d/%d] %s (%s)\n", done, hashN, idx.rel(l.path), human(l.size))
			sum, err := hashFile(ctx, l.path)
			if err != nil {
				if ctx.Err() != nil {
					st.saveOrWarn()
					return ls, ctx.Err()
				}
				fmt.Fprintf(os.Stderr, "  warning: %v\n", err)
				continue
			}
			e.SHA256 = sum
			st.saveOrWarn()
		}
	}

	from := len(res.items)
	for _, l := range todo {
		rel := idx.rel(l.path)
		e := st.Files[rel]
		var v *hashVersion
		why := ""
		switch {
		case e.VersionID != 0 && (!enabled || e.Base == "" || opt.previews == 0 || previewExists(l.path, opt.previews)):
			// Known from an earlier run and nothing to fetch: no need to ask.
			v = &hashVersion{apiVersion: apiVersion{ID: e.VersionID, BaseModel: e.BaseName, Name: e.VersionName}}
			v.Model.Name, v.Model.Type = e.Model, e.ModelType
		case !enabled:
			why = "not looked up on Civitai (-local does that)"
		case e.NotOnCivitai:
			why = "Civitai does not know it (nothing there has this checksum)"
		case e.SHA256 == "":
			why = "could not be checksummed"
		default:
			var err error
			v, err = c.versionByHash(ctx, e.SHA256)
			if ctx.Err() != nil {
				return ls, ctx.Err()
			}
			switch {
			case err != nil:
				why = "the Civitai lookup failed (" + err.Error() + "); the next run tries again"
				ls.failed++
			case v == nil:
				e.NotOnCivitai = true
				why = "Civitai does not know it (nothing there has this checksum)"
			}
		}
		tag, how := "", ""
		if v != nil {
			e.VersionID, e.ModelType, e.Model, e.BaseName = v.ID, v.Model.Type, v.Model.Name, strings.TrimSpace(v.BaseModel)
			if v.Name != "" {
				e.VersionName = v.Name
			}
			if v.Model.Name != "" {
				e.Title, e.Version = cleanTitle(v.Model.Name), e.VersionName
			}
			t := baseTag(v.BaseModel)
			if baseAgnostic[v.Model.Type] || noTag(t) {
				e.Base = ""
				why = fmt.Sprintf("Civitai has it (%s) but gives it no base model", v.Model.Name)
				v = nil
			} else {
				tag, how = t, "found on Civitai by checksum; not a liked model"
				e.Base = t
				ls.found++
			}
		}
		if tag == "" && e.Base != "" && e.VersionID == 0 && e.NotOnCivitai && !guess {
			e.Base = "" // a -guess from an earlier run, not wanted now
		}
		if tag == "" && guess && !strings.HasPrefix(why, "the Civitai lookup failed") {
			if g := guessBase(l.path); g != "" {
				tag, how = g, "base model from the file's own metadata (-guess)"
				ls.guessed++
			} else if why != "" {
				why += "; its metadata names no base model either"
			}
		}
		if tag != "" && !opt.all && !opt.bases[tag] {
			continue // another base model than -base asks for: not handled this run
		}
		if tag == "" {
			if !carriesTag(l.path, known) {
				ls.left = append(ls.left, leftFile{rel, why})
				if !enabled {
					ls.notLooked++
				}
			}
			continue
		}
		if res.names[tag] == "" {
			res.names[tag] = e.BaseName
		}
		knownTags[tag] = true
		name := strings.TrimSuffix(filepath.Base(l.path), filepath.Ext(l.path))
		it := item{tag: tag, status: stHave, found: l.path, local: true}
		it.model = apiModel{Name: name}
		if v != nil {
			it.model = apiModel{ID: v.ModelID, Name: v.Model.Name, Type: v.Model.Type}
			it.ver = v.apiVersion
			it.images = previewImages(v.apiVersion, nil)
		}
		it.note = rel + " (" + how + ")"
		current := untag(tag, filepath.Base(l.path))
		save, why := current, ""
		if v != nil {
			save, why = localSaveName(current, v.Model.Name, v.Name)
		}
		want := ggufHome(idx, desiredPath(l.path, tag, save))
		if !strings.EqualFold(want, l.path) {
			it.rename = want
			if why != "" {
				it.note = joinNote(it.note, "new name from the model: "+why)
			}
		}
		res.items = append(res.items, it)
	}
	st.saveOrWarn()
	settle(res, st, idx, opt, from)
	return ls, nil
}

// oddNames lists files whose name has a model extension in the middle, the
// usual result of a rename gone wrong: ComfyUI does not list them.
func oddNames(idx *index) []string {
	var out []string
	for _, l := range idx.all {
		name := strings.ToLower(filepath.Base(l.path))
		if isModelFile(l.path) {
			continue
		}
		for ext := range modelExts {
			i := strings.Index(name, ext)
			if i < 0 {
				continue
			}
			rest := name[i+len(ext):]
			if rest != "" && rest[0] != '.' { // ".part", ".json" and the like are fine
				out = append(out, idx.rel(l.path))
				break
			}
		}
	}
	sort.Strings(out)
	return out
}

// ---------------------------------------------------------------- download

type progress struct {
	total, done, start int64
	began, last        time.Time
	tty                bool
}

func (p *progress) tick(final bool) {
	now := time.Now()
	every := time.Second
	if !p.tty {
		every = 30 * time.Second
	}
	if !final && now.Sub(p.last) < every {
		return
	}
	p.last = now
	rate := float64(p.done-p.start) / math.Max(now.Sub(p.began).Seconds(), 0.001)
	line := fmt.Sprintf("    %s of %s  %s/s", human(p.done), human(p.total), human(int64(rate)))
	if p.total > 0 {
		line = fmt.Sprintf("    %3.0f%%  %s", 100*float64(p.done)/float64(p.total), strings.TrimSpace(line))
	}
	if p.tty {
		fmt.Fprintf(os.Stderr, "\r%-70s", line)
		if final {
			fmt.Fprintln(os.Stderr)
		}
	} else {
		fmt.Fprintln(os.Stderr, line)
	}
}

var (
	errChecksum = errors.New("checksum mismatch")
	errRestart  = errors.New("the partial file could not be resumed; starting over")
	errWall     = errors.New("Civitai sent a web page instead of the file (usually a login or access wall)")
)

// fetch downloads one file to dest via dest+".part", resuming a partial file.
func (c *client) fetch(ctx context.Context, it item, tty bool) (string, error) {
	part := it.dest + ".part"
	if err := os.MkdirAll(filepath.Dir(it.dest), 0o755); err != nil {
		return "", err
	}
	hasher := sha256.New()
	var offset int64
	if info, err := os.Stat(part); err == nil && info.Size() > 0 {
		f, err := os.Open(part)
		if err != nil {
			return "", err
		}
		offset, err = io.Copy(hasher, f)
		f.Close()
		if err != nil {
			return "", err
		}
	}

	dctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stall := time.AfterFunc(stallTimeout, cancel)
	defer stall.Stop()

	hdr := map[string]string{}
	if offset > 0 {
		hdr["Range"] = fmt.Sprintf("bytes=%d-", offset)
	}
	resp, err := c.do(dctx, it.file.DownloadURL, hdr)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	flags := os.O_CREATE | os.O_WRONLY
	switch {
	case resp.StatusCode == http.StatusPartialContent && offset > 0 &&
		strings.HasPrefix(resp.Header.Get("Content-Range"), fmt.Sprintf("bytes %d-", offset)):
		flags |= os.O_APPEND
	case resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusPartialContent:
		offset = 0 // server sent the whole file; start over
		hasher.Reset()
		flags |= os.O_TRUNC
	case resp.StatusCode == http.StatusRequestedRangeNotSatisfiable:
		os.Remove(part) // the retry starts clean
		return "", errRestart
	default:
		return "", statusError(resp)
	}
	if ct := resp.Header.Get("Content-Type"); strings.HasPrefix(ct, "text/html") {
		return "", errWall
	}

	out, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return "", err
	}
	total := it.file.bytes()
	if resp.ContentLength > 0 {
		total = offset + resp.ContentLength
	}
	pr := &progress{total: total, done: offset, start: offset, began: time.Now(), tty: tty}
	buf := make([]byte, 1<<20)
	var copyErr error
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			stall.Reset(stallTimeout)
			if _, werr := out.Write(buf[:n]); werr != nil {
				copyErr = werr
				break
			}
			hasher.Write(buf[:n])
			pr.done += int64(n)
			pr.tick(false)
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			copyErr = rerr
			break
		}
	}
	pr.tick(true)
	if cerr := out.Close(); copyErr == nil {
		copyErr = cerr
	}
	if copyErr != nil {
		if ctx.Err() == nil && dctx.Err() != nil {
			return "", fmt.Errorf("no data for %s; gave up on this attempt", stallTimeout)
		}
		return "", copyErr
	}
	if resp.ContentLength > 0 && pr.done != total {
		return "", fmt.Errorf("connection closed early (%s of %s)", human(pr.done), human(total))
	}

	got := hex.EncodeToString(hasher.Sum(nil))
	if want := it.file.sha256(); want != "" {
		if got != want {
			bad := it.dest + ".unverified"
			os.Remove(bad)
			os.Rename(part, bad)
			return "", fmt.Errorf("%w: kept as %s, not under its real name\n    %s", errChecksum, filepath.Base(bad),
				mismatchDetail(it, got, pr.done, resp.Header.Get("Content-Disposition")))
		}
	}
	if _, err := os.Lstat(it.dest); err == nil {
		os.Remove(part) // someone else put it there meanwhile; this copy is not needed
		return "", fmt.Errorf("%s appeared while downloading (another run?); kept that one", it.dest)
	}
	if err := os.Rename(part, it.dest); err != nil {
		return "", err
	}
	os.Remove(it.dest + ".unverified") // an earlier bad download of it, now replaced
	return got, nil
}

// mismatchDetail says what came instead of the file asked for: the name
// Civitai gave it, its size against the expected size, and whether it is in
// fact another file of the same model.
func mismatchDetail(it item, got string, size int64, disposition string) string {
	sent := ""
	if _, params, err := mime.ParseMediaType(disposition); err == nil {
		sent = params["filename"]
	}
	var parts []string
	if sent != "" && !strings.EqualFold(sent, it.file.Name) {
		parts = append(parts, fmt.Sprintf("Civitai sent %q when asked for %q", sent, it.file.Name))
	}
	for _, v := range it.model.ModelVersions {
		for _, f := range v.Files {
			if f.sha256() == got && f.Name != it.file.Name {
				parts = append(parts, fmt.Sprintf("what came is %q of version %q (Civitai's download link mixes these files up)", f.Name, v.Name))
			}
		}
	}
	want := it.file.bytes()
	switch {
	case want > 0 && !sizeClose(size, want):
		parts = append(parts, fmt.Sprintf("it is %s where Civitai lists %s", human(size), human(want)))
	case len(parts) == 0:
		parts = append(parts, "right size, wrong content: damaged on the way, or Civitai's checksum for it is out of date")
	}
	return strings.Join(parts, "; ")
}

func (c *client) fetchWithRetry(ctx context.Context, it item, tty bool) (string, error) {
	var last error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		sum, err := c.fetch(ctx, it, tty)
		if err == nil {
			return sum, nil
		}
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		last = err
		var he *httpError
		if errors.Is(err, errChecksum) || errors.Is(err, errWall) || (errors.As(err, &he) && !he.retryable()) {
			return "", err
		}
		if attempt < maxAttempts {
			wait := backoff(attempt, err)
			fmt.Fprintf(os.Stderr, "    %v; retrying in %s\n", err, wait)
			if err := sleepCtx(ctx, wait); err != nil {
				return "", err
			}
		}
	}
	return "", last
}

// -------------------------------------------------------------------- misc

func human(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.2f GB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.0f KB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%d B", n)
}

func freeBytes(dir string) (int64, bool) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return 0, false
	}
	return int64(st.Bavail) * int64(st.Bsize), true
}

func isTerminal(f *os.File) bool {
	info, err := f.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

func loadToken(file string) (string, error) {
	if t := strings.TrimSpace(os.Getenv("CIVITAI_TOKEN")); t != "" {
		return t, nil
	}
	data, err := os.ReadFile(file)
	if err == nil {
		if t := strings.TrimSpace(strings.SplitN(string(data), "\n", 2)[0]); t != "" {
			return t, nil
		}
	}
	return "", fmt.Errorf("no API token. Create one at https://civitai.com/user/account, then either\n"+
		"  export CIVITAI_TOKEN=...   or   put it in %s", file)
}

func printGroup(title string, items []item, status string, line func(item) string) int {
	n := 0
	for _, it := range items {
		if it.status == status {
			n++
		}
	}
	if n == 0 {
		return 0
	}
	fmt.Printf("\n%s (%d):\n", title, n)
	for _, it := range items {
		if it.status == status {
			fmt.Println("  " + line(it))
		}
	}
	return n
}

// label names a liked model in the report, with its base model tag.
func label(it item) string {
	if it.tag == "" {
		return it.model.Name
	}
	return it.model.Name + " [" + it.tag + "]"
}

func sortedKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// ------------------------------------------------------------------ config

// The config file is JSON. It holds:
//   - settings: the default of every flag (command-line flags win for that run;
//     -set-conf saves the ones given into the file);
//   - newBaseModels: what to do when a base model turns up in the likes for the
//     first time: "ask", "download" or "skip";
//   - bases: per base model, whether its versions are downloaded, optional
//     per-base previews and latest, and the liked models the tool has met with
//     what became of each. "download": false on a model keeps that model from
//     being downloaded for that base.
// The tool rewrites only its own parts (names, statuses, files, help); the
// choices are yours and are kept even when the file is edited during a run.

const noBaseKey = "none" // config key for models that have no base model (upscalers)

func baseKey(tag string) string {
	if tag == "" {
		return noBaseKey
	}
	return tag
}

func keyTag(key string) string {
	if key == noBaseKey {
		return ""
	}
	return key
}

type confModel struct {
	Name     string `json:"name"`
	Type     string `json:"type,omitempty"`
	Download bool   `json:"download"`
	Status   string `json:"status"`
	Version  string `json:"version,omitempty"`
	File     string `json:"file,omitempty"`
	Updated  string `json:"updated,omitempty"`
}

type confBase struct {
	Name     string                `json:"name"`
	Download *bool                 `json:"download"` // null: not chosen yet
	Previews *int                  `json:"previews"`
	Latest   *bool                 `json:"latest"`
	Models   map[string]*confModel `json:"models"` // by Civitai model id
}

type config struct {
	About         []string             `json:"about"`
	NewBaseModels string               `json:"newBaseModels"`
	Settings      map[string]any       `json:"settings"`
	Help          map[string]string    `json:"help"`
	Bases         map[string]*confBase `json:"bases"`

	path     string
	mtime    time.Time
	imported string // the old flag = value file this one was made from
}

// Flags that are not settings: they steer the config itself.
var notSettings = map[string]bool{"config": true, "print-config": true, "set-conf": true, "reconfig": true, "base": true, "version": true}

var configAbout = []string{
	"moFlo config. Flags on the command line win for that run; add -set-conf to save them here.",
	"settings: the default of every flag (see help). newBaseModels: \"ask\", \"download\" or \"skip\", for a base model that turns up in your likes for the first time.",
	"bases: per base model, \"download\" true or false (null: not chosen yet, asked about); \"previews\" and \"latest\" override the settings for that base (null = use the setting).",
	"bases.<base>.models: the liked models met so far, by Civitai model id. Set \"download\": false on one to stop it being downloaded for that base. The tool writes the rest.",
}

func defaultConfigPath(home string) string {
	if d, err := os.UserConfigDir(); err == nil {
		return filepath.Join(d, "moflo", "config.json")
	}
	return filepath.Join(home, ".config", "moflo", "config.json")
}

// legacyConfigDir is where the tool kept its config when it was called
// civitai-likes: config.json there, or the older flag = value file "config".
func legacyConfigDir(home string) string {
	return filepath.Join(filepath.Dir(filepath.Dir(defaultConfigPath(home))), "civitai-likes")
}

func loadConfig(path, legacyDir string) (*config, error) {
	c := &config{path: path, NewBaseModels: "ask"}
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		if err := json.Unmarshal(data, c); err != nil {
			return nil, configError(path, data, err)
		}
		if st, err := os.Stat(path); err == nil {
			c.mtime = st.ModTime()
		}
	case errors.Is(err, os.ErrNotExist) && legacyDir != "":
		old := filepath.Join(legacyDir, "config.json")
		if data, err := os.ReadFile(old); err == nil { // civitai-likes' JSON config: same format
			if err := json.Unmarshal(data, c); err != nil {
				return nil, configError(old, data, err)
			}
			c.imported = old
		} else if err := c.importLegacy(filepath.Join(legacyDir, "config")); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("config %s: %v", path, err)
	}
	return c, c.normalize()
}

func (c *config) normalize() error {
	if c.Settings == nil {
		c.Settings = map[string]any{}
	}
	if c.Bases == nil {
		c.Bases = map[string]*confBase{}
	}
	for k, b := range c.Bases {
		if b == nil {
			b = &confBase{}
			c.Bases[k] = b
		}
		if b.Models == nil {
			b.Models = map[string]*confModel{}
		}
		for id, m := range b.Models {
			if m == nil {
				delete(b.Models, id)
			}
		}
	}
	c.NewBaseModels = strings.ToLower(strings.TrimSpace(c.NewBaseModels))
	if c.NewBaseModels == "" {
		c.NewBaseModels = "ask"
	}
	if c.NewBaseModels != "ask" && c.NewBaseModels != "download" && c.NewBaseModels != "skip" {
		return fmt.Errorf("config %s: newBaseModels must be \"ask\", \"download\" or \"skip\", not %q", c.path, c.NewBaseModels)
	}
	return nil
}

// configError says where in the file a JSON mistake is.
func configError(path string, data []byte, err error) error {
	var off int64 = -1
	var se *json.SyntaxError
	var te *json.UnmarshalTypeError
	switch {
	case errors.As(err, &se):
		off = se.Offset
	case errors.As(err, &te):
		off = te.Offset
		err = fmt.Errorf("%s should be %s, not a JSON %s", te.Field, te.Type, te.Value)
	}
	if off < 0 {
		return fmt.Errorf("config %s: %v", path, err)
	}
	line, col := 1, 1
	for _, b := range data[:min(int(off), len(data))] {
		if b == '\n' {
			line, col = line+1, 1
		} else {
			col++
		}
	}
	return fmt.Errorf("config %s line %d column %d: %v", path, line, col, err)
}

// importLegacy reads an old "flag = value" config file.
func (c *config) importLegacy(path string) error {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("config %s: %v", path, err)
	}
	c.Settings = map[string]any{}
	for n, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		key = strings.TrimLeft(strings.TrimSpace(key), "-")
		val = strings.Trim(strings.TrimSpace(val), `"'`)
		if !ok {
			key, val = strings.TrimLeft(line, "-"), "true"
		}
		if key == "base" {
			set, all := parseBases(val)
			if all {
				c.NewBaseModels = "download"
			}
			for t := range set {
				c.base(baseKey(t)).Download = boolp(true)
			}
			continue
		}
		if notSettings[key] {
			continue
		}
		f := flag.Lookup(key)
		if f == nil {
			return fmt.Errorf("config %s line %d: no flag called %q", path, n+1, key)
		}
		v, err := typedValue(f, val)
		if err != nil {
			return fmt.Errorf("config %s line %d: %s = %s: %v", path, n+1, key, val, err)
		}
		c.Settings[key] = v
	}
	c.imported = path
	return nil
}

// typedValue turns a flag value written as text into the JSON type of that flag.
func typedValue(f *flag.Flag, s string) (any, error) {
	switch f.Value.(flag.Getter).Get().(type) {
	case bool:
		return strconv.ParseBool(s)
	case int:
		return strconv.Atoi(s)
	}
	return s, nil
}

func settingText(v any) (string, bool) {
	switch v := v.(type) {
	case bool:
		return strconv.FormatBool(v), true
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64), true
	case int:
		return strconv.Itoa(v), true
	case string:
		return v, true
	}
	return "", false
}

// applySettings sets every flag the settings name, except those given on the
// command line.
func (c *config) applySettings(onCLI map[string]bool) error {
	for _, k := range slices.Sorted(maps.Keys(c.Settings)) {
		v := c.Settings[k]
		if v == nil {
			continue // null: the built-in default
		}
		if notSettings[k] {
			return fmt.Errorf("config %s: %q is not a setting (it only works on the command line)", c.path, k)
		}
		f := flag.Lookup(k)
		if f == nil {
			return fmt.Errorf("config %s: settings has %q, but there is no such flag (moflo -h lists them)", c.path, k)
		}
		if onCLI[k] {
			continue
		}
		s, ok := settingText(v)
		if !ok {
			return fmt.Errorf("config %s: settings %q should be true/false, a number or a string", c.path, k)
		}
		if err := flag.Set(k, s); err != nil {
			return fmt.Errorf("config %s: settings %q = %v: %v", c.path, k, v, err)
		}
	}
	return nil
}

// setFromCLI saves the flags given on the command line as settings (-set-conf).
// -n is never saved: a file that makes every run a dry run helps nobody.
func (c *config) setFromCLI(onCLI map[string]bool) []string {
	var saved []string
	flag.Visit(func(f *flag.Flag) { // flag.Visit also sees what the config set: only take the command line's
		if !onCLI[f.Name] || notSettings[f.Name] || f.Name == "n" {
			return
		}
		c.Settings[f.Name] = f.Value.(flag.Getter).Get()
		saved = append(saved, fmt.Sprintf("%s = %v", f.Name, f.Value.String()))
	})
	return saved
}

// fill adds every setting the file lacks at its built-in default, and the help.
func (c *config) fill() {
	c.About = configAbout
	c.Help = map[string]string{}
	flag.VisitAll(func(f *flag.Flag) {
		if notSettings[f.Name] {
			return
		}
		c.Help[f.Name] = strings.Join(strings.Fields(f.Usage), " ")
		if _, ok := c.Settings[f.Name]; !ok {
			if v, err := typedValue(f, f.DefValue); err == nil {
				c.Settings[f.Name] = v
			}
		}
	})
}

func boolp(b bool) *bool { return &b }

func (b *confBase) on() bool { return b.Download != nil && *b.Download }

func (c *config) base(key string) *confBase {
	b := c.Bases[key]
	if b == nil {
		b = &confBase{Models: map[string]*confModel{}}
		if c.Bases == nil {
			c.Bases = map[string]*confBase{}
		}
		c.Bases[key] = b
	}
	return b
}

// save writes the file. When it was changed on disk since it was read (edited
// during a run), the choices in it are taken over first, so none are lost.
func (c *config) save() error {
	if st, err := os.Stat(c.path); err == nil && !st.ModTime().Equal(c.mtime) {
		if data, err := os.ReadFile(c.path); err == nil {
			disk := &config{path: c.path}
			if json.Unmarshal(data, disk) == nil && disk.normalize() == nil {
				c.adopt(disk)
			}
		}
	}
	c.fill()
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(c.path), 0o755); err != nil {
		return err
	}
	tmp := c.path + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, c.path); err != nil {
		return err
	}
	if st, err := os.Stat(c.path); err == nil {
		c.mtime = st.ModTime()
	}
	return nil
}

func (c *config) saveOrWarn() {
	if err := c.save(); err != nil {
		fmt.Fprintln(os.Stderr, "warning: could not save", c.path+":", err)
	}
}

// adopt takes the user's choices from the copy on disk.
func (c *config) adopt(disk *config) {
	c.Settings, c.NewBaseModels = disk.Settings, disk.NewBaseModels
	for k, db := range disk.Bases {
		mb := c.Bases[k]
		if mb == nil {
			c.Bases[k] = db
			continue
		}
		mb.Download, mb.Previews, mb.Latest = db.Download, db.Previews, db.Latest
		for id, dm := range db.Models {
			if mm := mb.Models[id]; mm != nil {
				mm.Download = dm.Download
			} else {
				mb.Models[id] = dm
			}
		}
	}
}

// effective prints the config as it applies to this run (-print-config).
func (c *config) effective(onCLI map[string]bool) ([]byte, error) {
	cp := *c
	cp.Settings = maps.Clone(c.Settings)
	cp.fill()
	flag.Visit(func(f *flag.Flag) {
		if onCLI[f.Name] && !notSettings[f.Name] {
			cp.Settings[f.Name] = f.Value.(flag.Getter).Get()
		}
	})
	return json.MarshalIndent(&cp, "", "  ")
}

// ------------------------------------------------------- choosing base models

type likedBase struct {
	key, name string
	count     int // liked models with a version for it
}

// likedBases lists the base models in the likes, most liked first.
func likedBases(models []apiModel) []likedBase {
	by := map[string]*likedBase{}
	for _, m := range models {
		if _, ok := typeDirs[m.Type]; !ok {
			continue
		}
		seen := map[string]bool{}
		for _, v := range m.ModelVersions {
			key, name := noBaseKey, "No base model (upscalers)"
			if !baseAgnostic[m.Type] {
				key, name = baseKey(baseTag(v.BaseModel)), strings.TrimSpace(v.BaseModel)
			}
			if seen[key] {
				continue
			}
			seen[key] = true
			lb := by[key]
			if lb == nil {
				lb = &likedBase{key: key, name: name}
				by[key] = lb
			}
			lb.count++
		}
	}
	out := make([]likedBase, 0, len(by))
	for _, lb := range by {
		out = append(out, *lb)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].count != out[j].count {
			return out[i].count > out[j].count
		}
		return out[i].key < out[j].key
	})
	return out
}

var stdin = bufio.NewReader(os.Stdin)

// parsePicks reads "1 2 3", "1,2,3", "2-5", "all" or "none" against n choices.
func parsePicks(answer string, n int) (map[int]bool, error) {
	picks := map[int]bool{}
	for _, tok := range strings.FieldsFunc(answer, func(r rune) bool { return r == ',' || r == ' ' || r == '\t' }) {
		switch strings.ToLower(tok) {
		case "all":
			for i := 1; i <= n; i++ {
				picks[i] = true
			}
			continue
		case "none":
			continue
		}
		lo, hi, isRange := strings.Cut(tok, "-")
		a, err1 := strconv.Atoi(lo)
		b, err2 := a, error(nil)
		if isRange {
			b, err2 = strconv.Atoi(hi)
		}
		if err1 != nil || err2 != nil || a < 1 || b > n || a > b {
			return nil, fmt.Errorf("%q is not a number from 1 to %d", tok, n)
		}
		for i := a; i <= b; i++ {
			picks[i] = true
		}
	}
	return picks, nil
}

// chooseBases makes sure every base model in the likes has a download choice.
// The first time, for new ones when newBaseModels is "ask", and with -reconfig,
// it lists them and asks. It reports whether anything changed.
func (c *config) chooseBases(liked []likedBase, reconfig, noAsk bool) (bool, error) {
	first := true
	for _, b := range c.Bases {
		if b.Download != nil {
			first = false
		}
	}
	var fresh []string
	list := append([]likedBase(nil), liked...)
	inLikes := map[string]bool{}
	for _, lb := range liked {
		inLikes[lb.key] = true
		b := c.base(lb.key)
		if b.Download == nil {
			fresh = append(fresh, lb.key)
		}
		if lb.name != "" {
			b.Name = lb.name
		}
	}
	for _, k := range slices.Sorted(maps.Keys(c.Bases)) { // chosen before, no longer in the likes
		if !inLikes[k] {
			list = append(list, likedBase{key: k, name: c.Bases[k].Name})
		}
	}
	isFresh := func(k string) bool { return slices.Contains(fresh, k) }
	if !reconfig && len(fresh) == 0 {
		return false, nil
	}
	if !reconfig && !first && c.NewBaseModels != "ask" {
		for _, k := range fresh {
			c.Bases[k].Download = boolp(c.NewBaseModels == "download")
		}
		fmt.Printf("New base model(s) in your likes: %s; %s, as newBaseModels says.\n",
			strings.Join(fresh, ", "), map[bool]string{true: "downloading them", false: "not downloading them"}[c.NewBaseModels == "download"])
		return true, nil
	}
	if noAsk {
		if reconfig {
			return false, errors.New("-reconfig asks which base models to download; leave out -yes")
		}
		for _, k := range fresh {
			c.Bases[k].Download = boolp(first)
		}
		if first {
			fmt.Println("First run with -yes: every base model in your likes is set to download. Pick with -reconfig,")
			fmt.Printf("or set \"download\" under each base model in %s.\n", c.path)
		} else {
			fmt.Printf("New base model(s) in your likes, not downloaded (-yes cannot ask): %s\n", strings.Join(fresh, ", "))
			fmt.Printf("Turn one on with -reconfig, or \"download\": true under it in %s.\n", c.path)
		}
		return true, nil
	}

	switch {
	case first:
		fmt.Println("\nBase models in your likes. Which should be downloaded?")
	case reconfig:
		fmt.Println("\nBase models (x = downloaded now). Which should be downloaded?")
	default:
		fmt.Println("\nNew base models turned up in your likes (marked new). Which should be downloaded?")
	}
	for i, lb := range list {
		mark := "   "
		if c.Bases[lb.key].on() {
			mark = "[x]"
		}
		extra := fmt.Sprintf("%3d liked", lb.count)
		if !inLikes[lb.key] {
			extra = "  0 liked"
		}
		if isFresh(lb.key) && !first {
			extra += "  new"
		}
		name := lb.name
		if name == "" {
			name = lb.key
		}
		fmt.Printf("  %s %2d  %-26s %-16s %s\n", mark, i+1, name, lb.key, extra)
	}
	enter := "keep as marked"
	if first {
		enter = "all"
	}
	var picks map[int]bool
	for tries := 0; ; tries++ {
		fmt.Printf("Numbers like 1 2 3 or 1,2,3 (ranges like 2-5, all, none) [Enter = %s]: ", enter)
		answer, err := stdin.ReadString('\n')
		answer = strings.TrimSpace(answer)
		if answer == "" {
			if err != nil {
				fmt.Println()
			}
			break // keep as marked (first run: all)
		}
		p, perr := parsePicks(answer, len(list))
		if perr == nil {
			picks = p
			break
		}
		fmt.Println("  " + perr.Error())
		if err != nil || tries >= 4 {
			return false, errors.New("no usable answer; nothing chosen")
		}
	}
	for i, lb := range list {
		b := c.Bases[lb.key]
		switch {
		case picks != nil:
			b.Download = boolp(picks[i+1])
		case first:
			b.Download = boolp(true)
		case isFresh(lb.key):
			b.Download = boolp(false)
		}
	}
	var on []string
	for _, lb := range list {
		if c.Bases[lb.key].on() {
			on = append(on, lb.key)
		}
	}
	if len(on) == 0 {
		on = []string{"none"}
	}
	fmt.Printf("Downloading: %s\n", strings.Join(on, ", "))
	fmt.Printf("Saved to %s. Change it there (\"download\" under each base model) or run with -reconfig.\n\n", c.path)
	return true, nil
}

// enabled is the set of base model tags to download ("" for no base model).
func (c *config) enabled() map[string]bool {
	set := map[string]bool{}
	for k, b := range c.Bases {
		if b.on() {
			set[keyTag(k)] = true
		}
	}
	return set
}

// ------------------------------------------------------ what became of models

func modelKey(tag string, id int64) string { return baseKey(tag) + "/" + strconv.FormatInt(id, 10) }

// blocked lists the models set to "download": false, by modelKey.
func (c *config) blocked() map[string]bool {
	out := map[string]bool{}
	for k, b := range c.Bases {
		for id, m := range b.Models {
			if !m.Download {
				out[k+"/"+id] = true
			}
		}
	}
	return out
}

// note records what became of one liked model for one base model.
func (c *config) note(it item, status, file string) {
	if it.local || it.model.ID == 0 {
		return
	}
	if it.tag == "" && !baseAgnostic[it.model.Type] {
		return // not a model ComfyUI loads, or nothing downloadable: no base to file it under
	}
	b := c.base(baseKey(it.tag))
	id := strconv.FormatInt(it.model.ID, 10)
	m := b.Models[id]
	if m == nil {
		m = &confModel{Download: true}
		b.Models[id] = m
	}
	m.Name, m.Type = it.model.Name, it.model.Type
	if it.ver.Name != "" {
		m.Version = it.ver.Name
	}
	if file != "" {
		m.File = file
	}
	if m.Status != status {
		m.Status = status
		m.Updated = time.Now().Format("2006-01-02")
	}
}

// statusOf says in words what the plan is for an item.
func statusOf(it item) string {
	switch it.status {
	case stHave:
		return "on disk"
	case stGet:
		if it.ea {
			return "to download (early access until " + eaUntil(it.ver.EarlyAccessEndsAt) + ")"
		}
		return "to download"
	case stWait:
		return "held back: " + it.note
	case stAlone:
		return "left alone: " + it.note
	case stOff:
		return "base model not downloaded"
	}
	return "skipped: " + it.note
}

// noteAll records the plan for every liked model, and marks the models met
// before that are no longer liked.
func (c *config) noteAll(items []item, models []apiModel, idx *index) {
	for _, it := range items {
		file := ""
		if it.status == stHave {
			file = idx.rel(it.found)
		}
		c.note(it, statusOf(it), file)
	}
	liked := map[string]bool{}
	for _, m := range models {
		liked[strconv.FormatInt(m.ID, 10)] = true
	}
	for _, b := range c.Bases {
		for id, m := range b.Models {
			if !liked[id] && m.Status != "no longer liked" {
				m.Status, m.Updated = "no longer liked", time.Now().Format("2006-01-02")
			}
		}
	}
}

// baseTally counts what a run found for one base model's liked files.
type baseTally struct{ have, get, wait, alone, off int }

func tallyBases(items []item) map[string]*baseTally {
	per := map[string]*baseTally{}
	for _, it := range items {
		if it.local || (it.tag == "" && !baseAgnostic[it.model.Type]) {
			continue
		}
		c := per[it.tag]
		if c == nil {
			c = &baseTally{}
			per[it.tag] = c
		}
		switch it.status {
		case stHave:
			c.have++
		case stGet:
			c.get++
		case stWait:
			c.wait++
		case stAlone:
			c.alone++
		case stSkip:
			if strings.Contains(it.note, "in the config") {
				c.off++
			}
		}
	}
	return per
}

func run() int {
	home, _ := os.UserHomeDir()
	modelsDir := flag.String("models", filepath.Join(home, "ComfyUI", "models"), "ComfyUI models folder")
	dryRun := flag.Bool("n", false, "show what would happen; change nothing")
	yes := flag.Bool("yes", false, "go ahead without asking")
	baseSpec := flag.String("base", "all", "base models to handle (download, rename, preview), comma separated, as tags\n"+
		"(flux1_dev) or Civitai names (\"Flux.1 D\"); \"all\" (the default) handles every base")
	rename := flag.Bool("rename", false, "rename files you already have so their names start with the base model tag")
	previews := flag.Int("previews", 1, "preview images to keep per model file; the first is the one ComfyUI shows,\n"+
		"extras are saved as <name>.preview-2.jpg and so on for you to choose from (0 = none)")
	renameOnly := flag.Bool("rename-only", false, "only rename model files you already have to the tagged form (liked models of any\n"+
		"base; with -local, other files too)\n"+
		"(their previews and info files go with them); no downloads, no previews")
	previewsOnly := flag.Bool("previews-only", false, "only fetch missing previews for liked models you already have, up to\n"+
		"-previews per file; no model downloads or renames")
	previewInfo := flag.Bool("preview-info", false, "also save the generation settings (prompt, sampler, seed...) of preview pictures\n"+
		"already on disk that have none, matching each to Civitai's images byte for byte\n"+
		"(those images are downloaded again to compare). New previews always get them")
	eaRetry := flag.Bool("ea", false, "retry early access versions Civitai refused before, without waiting for their\n"+
		"early access to end (for ones you have since paid for); on its own, only those")
	eaAll := flag.Bool("all", false, "with -ea: also do everything a normal run does")
	checksum := flag.Bool("checksum", false, "checksum every model file that has no checksum on record (slow, once per file).\n"+
		"Without it, downloads, -local files, and files that could be a liked file or a copy\n"+
		"of another are checksummed; that is enough to catch copies and tell files apart")
	local := flag.Bool("local", false, "also look up model files that are not liked models on Civitai by checksum, so they\n"+
		"get tagged and previewed too (the first run checksums them all, which is slow)")
	guess := flag.Bool("guess", false, "tag model files Civitai does not know (your own trainings, files from elsewhere)\n"+
		"by the base model their own metadata names, when it names one")
	latest := flag.Bool("latest", false, "fetch the newest version even when an older version is on disk")
	maxModels := flag.Int("max", 1000, "refuse to act on a likes list longer than this")
	stateOpt := flag.String("state", "", "state file (default <models>/.moflo.json)")
	tokenFile := flag.String("token-file", filepath.Join(home, ".config", "civitai", "token"), "file holding the API token")
	apiBase := flag.String("api", "https://civitai.com", "Civitai base URL")
	nameFlag := flag.String("name", "auto", "what downloaded model files are called:\n"+
		"  auto     Civitai's file name, unless it says nothing (ComfyUI1, lora, a UUID, a number)\n"+
		"           or another liked model uses it too: then the model's name and version\n"+
		"  civitai  Civitai's file name, always\n"+
		"  model    the model's name and version, always (with -rename, files you have are\n"+
		"           renamed too; saved ComfyUI workflows refer to files by name)")
	tagFlag := flag.String("tag", "front", "where the base model tag goes in file names:\n"+
		"  front   flux1_dev_name.safetensors\n"+
		"  end     name_flux1_dev.safetensors\n"+
		"  dot     name.flux1_dev.safetensors\n"+
		"  folder  loras/flux1_dev/name.safetensors\n"+
		"  none    Civitai's name, untouched")
	configPath := flag.String("config", defaultConfigPath(home), "config file (JSON): flag defaults, base models to download, and per base\n"+
		"model the liked models with what became of them. Command-line flags win for that run")
	printConfig := flag.Bool("print-config", false, "print the config as it applies to this run, and exit")
	showVersion := flag.Bool("version", false, "print the version and exit")
	setConf := flag.Bool("set-conf", false, "save the flags given on the command line into the config file as the new\n"+
		"defaults (-base sets which base models are downloaded); -n is never saved")
	reconfig := flag.Bool("reconfig", false, "list the base models in your likes again and ask which to download")
	flag.Parse()
	if *showVersion {
		fmt.Println("moflo", version)
		return 0
	}
	onCLI := map[string]bool{}
	flag.Visit(func(f *flag.Flag) { onCLI[f.Name] = true })
	legacy := ""
	if !onCLI["config"] {
		legacy = legacyConfigDir(home)
	}
	cfg, err := loadConfig(*configPath, legacy)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	if err := cfg.applySettings(onCLI); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	if *printConfig {
		data, err := cfg.effective(onCLI)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 2
		}
		fmt.Println(string(data))
		return 0
	}
	if cfg.imported != "" {
		if err := cfg.save(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 2
		}
		fmt.Printf("Imported %s into %s; the old file is no longer read (delete it when you like).\n", cfg.imported, cfg.path)
	}
	if *setConf {
		saved := cfg.setFromCLI(onCLI)
		if onCLI["base"] {
			set, all := parseBases(*baseSpec)
			if all {
				cfg.NewBaseModels = "download"
			}
			for _, b := range cfg.Bases {
				b.Download = boolp(all)
			}
			for t := range set {
				cfg.base(baseKey(t)).Download = boolp(true)
			}
			saved = append(saved, "base = "+*baseSpec)
		}
		if err := cfg.save(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 2
		}
		if len(saved) == 0 {
			fmt.Println("-set-conf: no flags given to save.")
		} else {
			fmt.Printf("Saved to %s: %s\n", cfg.path, strings.Join(saved, ", "))
		}
	}
	nameMode = strings.ToLower(strings.TrimSpace(*nameFlag))
	if !slices.Contains(nameModes, nameMode) {
		fmt.Fprintf(os.Stderr, "-name must be one of %s\n", strings.Join(nameModes, ", "))
		return 2
	}
	tagMode = strings.ToLower(strings.TrimSpace(*tagFlag))
	if !slices.Contains(tagModes, tagMode) {
		fmt.Fprintf(os.Stderr, "-tag must be one of %s\n", strings.Join(tagModes, ", "))
		return 2
	}
	if flag.NArg() > 0 {
		fmt.Fprintln(os.Stderr, "unexpected argument:", flag.Arg(0))
		return 2
	}
	if *previews < 0 {
		*previews = 0
	}
	if *eaAll && !*eaRetry {
		fmt.Fprintln(os.Stderr, "note: -all only means something together with -ea; ignored")
		*eaAll = false
	}
	eaOnly := *eaRetry && !*eaAll
	if *renameOnly {
		if *previewsOnly || *eaRetry {
			fmt.Fprintln(os.Stderr, "-rename-only does not go together with -previews-only or -ea")
			return 2
		}
		*rename = true
		*previews = 0
	}
	if *previewsOnly && *eaRetry {
		fmt.Fprintln(os.Stderr, "-previews-only and -ea do not go together")
		return 2
	}
	if eaOnly && *rename {
		fmt.Fprintln(os.Stderr, "note: -rename is ignored with -ea on its own (add -all for a full run)")
		*rename = false
	}
	if *previewsOnly {
		if *previews == 0 && !*previewInfo {
			fmt.Fprintln(os.Stderr, "-previews-only with -previews 0 has nothing to do")
			return 2
		}
		if *rename {
			fmt.Fprintln(os.Stderr, "note: -rename is ignored with -previews-only")
			*rename = false
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	token, err := loadToken(*tokenFile)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	base, err := url.Parse(strings.TrimRight(*apiBase, "/"))
	if err != nil || base.Host == "" {
		fmt.Fprintln(os.Stderr, "bad -api URL")
		return 2
	}
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.ResponseHeaderTimeout = apiTimeout
	c := &client{base: base, token: token, http: &http.Client{Transport: tr}}

	root, err := filepath.Abs(*modelsDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	if lock, err := lockModels(root); err != nil {
		if !*dryRun {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		fmt.Fprintln(os.Stderr, "note:", err, "(a dry run goes ahead; its plan may be out of date)")
	} else {
		defer lock.Close()
	}
	statePath := *stateOpt
	if statePath == "" {
		statePath = filepath.Join(root, ".moflo.json")
		// Renamed from civitai-likes: carry its state file over once.
		old := filepath.Join(root, ".civitai-likes.json")
		if _, err := os.Lstat(statePath); errors.Is(err, os.ErrNotExist) {
			if _, err := os.Lstat(old); err == nil {
				if err := os.Rename(old, statePath); err != nil {
					fmt.Fprintln(os.Stderr, "could not move the old state file:", err)
					return 2
				}
				fmt.Printf("Moved %s to %s (the tool is now called moFlo).\n", old, statePath)
			}
		}
	}
	if statePath, err = filepath.Abs(statePath); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	st, err := loadState(statePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}

	user, err := c.whoami(ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Civitai sign-in failed:", err)
		return 1
	}
	if user != "" {
		fmt.Println("Civitai account:", user)
	}
	models, err := c.likedModels(ctx, *maxModels)
	if err != nil {
		fmt.Fprintln(os.Stderr, "could not get the liked models:", err)
		return 1
	}
	if !onCLI["base"] { // which base models: from the config, asking about new ones
		changed, err := cfg.chooseBases(likedBases(models), *reconfig, *yes)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 2
		}
		if changed {
			if err := cfg.save(); err != nil {
				fmt.Fprintln(os.Stderr, err)
				return 2
			}
		}
	} else if *reconfig {
		fmt.Fprintln(os.Stderr, "note: -reconfig is ignored with -base")
	}

	scanStart := time.Now()
	idx, err := buildIndex(root, map[string]bool{statePath: true, statePath + ".tmp": true})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	if err := syncState(ctx, idx, st, *checksum); err != nil {
		fmt.Fprintln(os.Stderr, "\nInterrupted. Checksums done so far are saved.")
		return 130
	}
	// The scan is recorded now, so files that turn up from here on count as
	// new next time. Everything this run puts in place is recorded as it goes.
	st.LastScan = scanStart
	st.saveOrWarn()

	opt := planOptions{latest: *latest, rename: *rename, previews: *previews,
		blocked: cfg.blocked(), basePreviews: map[string]int{}, baseLatest: map[string]bool{}}
	if onCLI["base"] {
		opt.bases, opt.all = parseBases(*baseSpec)
	} else {
		opt.bases = cfg.enabled()
	}
	for k, b := range cfg.Bases { // per base model settings, unless the flag was given for this run
		if b.Previews != nil && !onCLI["previews"] && !*renameOnly {
			opt.basePreviews[keyTag(k)] = max(*b.Previews, 0)
		}
		if b.Latest != nil && !onCLI["latest"] {
			opt.baseLatest[keyTag(k)] = *b.Latest
		}
	}
	fmt.Printf("Liked models: %d\n", len(models))
	fmt.Printf("Files under %s: %d (%s)\n", root, len(idx.all), human(idx.bytes))
	if opt.all {
		fmt.Println("Base models: all")
	} else {
		var on []string
		for _, t := range sortedKeys(opt.bases) {
			on = append(on, baseKey(t))
		}
		if len(on) == 0 {
			on = []string{"none chosen (-reconfig to pick)"}
		}
		fmt.Println("Base models:", strings.Join(on, ", "))
	}
	if !st.existed {
		fmt.Println("First run: file sizes and dates recorded. Nothing is checksummed except downloads;")
		fmt.Println("-checksum checksums what you have (slow, once per file) to catch copies under other names.")
	}

	for t := range opt.bases {
		if t != "" {
			knownTags[t] = true
		}
	}
	for _, m := range models {
		for _, v := range m.ModelVersions {
			if t := baseTag(v.BaseModel); !noTag(t) && v.BaseModel != "" {
				knownTags[t] = true
			}
		}
	}
	// Copies of one file (two runs with different -tag, say) and partial
	// downloads someone else finished: cleaned up, and left out of the plan.
	// Checksums moFlo wants on its own (-checksum has done the rest already):
	// asked first, with the reason for each file.
	jobs := copyCandidates(idx)
	ambJobs, unsure := ambiguousCandidates(idx, models)
	at := map[string]int{}
	for i, j := range jobs {
		at[j.l.path] = i
	}
	for _, j := range ambJobs {
		if i, ok := at[j.l.path]; ok {
			jobs[i].why += "\n      and " + j.why
		} else {
			jobs = append(jobs, j)
		}
	}
	sort.Slice(jobs, func(i, j int) bool { return jobs[i].l.path < jobs[j].l.path })
	if len(jobs) > 0 && askChecksums(idx, jobs, *yes) {
		if err := hashJobs(ctx, idx, st, jobs); err != nil {
			fmt.Fprintln(os.Stderr, "\nInterrupted. Checksums done so far are saved.")
			return 130
		}
	}
	copies := groupCopies(idx)
	stale := staleParts(idx)
	var cleanBytes int64
	gone := map[string]bool{}
	for _, g := range copies {
		for _, d := range g.drop {
			gone[d.path] = true
			cleanBytes += d.size
		}
	}
	for _, p := range stale {
		gone[p.path] = true
		cleanBytes += p.size
	}
	idx.drop(gone)
	cleanN := len(gone)
	opt.removing, opt.links = gone, linkTargets(idx)
	opt.unsure = map[string]bool{}
	for p := range unsure {
		if idx.shaByPath[p] == "" && !gone[p] {
			opt.unsure[p] = true
		}
	}
	res := plan(c, idx, st, models, opt)
	ls, err := lookupLocal(ctx, c, idx, st, res, opt, *local && !eaOnly, *guess && !eaOnly)
	if err != nil {
		if ctx.Err() != nil {
			fmt.Fprintln(os.Stderr, "\nInterrupted. Checksums done so far are saved.")
			return 130
		}
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	for _, it := range res.items {
		if it.status == stHave {
			st.setBase(idx, it.found, it, res.names[it.tag])
		}
	}
	st.saveOrWarn()
	tallies := tallyBases(res.items) // before anything is filtered out of the list
	if !*dryRun {
		cfg.noteAll(slices.Concat(res.items, res.off), models, idx)
	}
	items := res.items
	notOnDisk := 0
	if *previewsOnly || *renameOnly {
		// Nothing gets downloaded, so only files already on disk matter.
		kept := items[:0]
		for _, it := range items {
			if it.status == stGet {
				notOnDisk++
				continue
			}
			kept = append(kept, it)
		}
		items = kept
	}
	notRetried := 0
	if !*previewsOnly && !*renameOnly {
		applyEarlyAccess(ctx, c, st, items, *eaRetry, time.Now())
		if ctx.Err() != nil {
			fmt.Fprintln(os.Stderr, "\nInterrupted.")
			return 130
		}
		tallies = tallyBases(items) // now with what is held back
		if !*dryRun {
			for _, it := range items {
				if it.status == stWait || it.status == stGet {
					cfg.note(it, statusOf(it), "")
				}
			}
		}

	}
	if eaOnly {
		// Only the early access versions refused before; everything else waits
		// for a normal run (or -ea -all).
		kept := items[:0]
		for _, it := range items {
			switch {
			case it.status == stGet && !it.eaBlocked:
				notRetried++
				continue
			case it.status == stHave:
				it.thumbs, it.rename = nil, ""
			}
			kept = append(kept, it)
		}
		items = kept
	}
	if !*dryRun {
		cfg.saveOrWarn() // the plan for every liked model, before anything happens
	}
	sort.SliceStable(items, func(i, j int) bool {
		a, b := strings.ToLower(items[i].model.Name), strings.ToLower(items[j].model.Name)
		if a != b {
			return a < b
		}
		return items[i].tag < items[j].tag
	})

	untagged := 0
	for _, it := range items {
		if it.status == stHave && it.rename != "" {
			untagged++
		}
	}
	infoPics, infoFiles := 0, 0
	if *previewInfo && !eaOnly {
		for i := range items {
			it := &items[i]
			if it.status != stHave || len(it.images) == 0 {
				continue
			}
			for _, pic := range previewPictures(it.found) {
				if _, err := os.Lstat(infoPath(pic)); err != nil {
					it.noInfo = append(it.noInfo, pic)
				}
			}
			if len(it.noInfo) > 0 {
				infoPics += len(it.noInfo)
				infoFiles++
			}
		}
	}
	if *renameOnly {
		haveN, blocked := 0, 0
		for _, it := range items {
			if it.status == stHave {
				haveN++
				if strings.Contains(it.note, "cannot be renamed") {
					blocked++
				}
			}
		}
		fmt.Printf("\nRename only. Liked model files on disk: %d", haveN-ls.found)
		if notOnDisk > 0 {
			fmt.Printf("; not on disk, so skipped: %d", notOnDisk)
		}
		fmt.Println(".")
		if blocked > 0 {
			fmt.Printf("\nCannot be renamed (%d):\n", blocked)
			for _, it := range items {
				if it.status == stHave && strings.Contains(it.note, "cannot be renamed") {
					fmt.Printf("  %s  ->  %s\n", label(it), it.note)
				}
			}
		}
	}
	if *renameOnly || *previewsOnly {
		var unsure []string
		for _, it := range items {
			if it.status == stAlone && strings.Contains(it.note, "-checksum tells") {
				unsure = append(unsure, fmt.Sprintf("  %s: %s", label(it), it.note))
			}
		}
		if len(unsure) > 0 {
			fmt.Printf("\nLeft alone, not sure which liked file they are (%d):\n%s\n", len(unsure), strings.Join(unsure, "\n"))
		}
	}
	if *renameOnly {
	} else if *previewsOnly {
		haveN := 0
		for _, it := range items {
			if it.status == stHave {
				haveN++
			}
		}
		fmt.Printf("\nPreviews only, up to %d per file. Liked model files on disk: %d", *previews, haveN-ls.found)
		if notOnDisk > 0 {
			fmt.Printf("; not on disk, so skipped: %d", notOnDisk)
		}
		fmt.Println(".")
	} else {
		printGroup("Already have", items, stHave, func(it item) string {
			return fmt.Sprintf("%s  ->  %s", label(it), it.note)
		})
		printGroup("Left alone", items, stAlone, func(it item) string {
			return fmt.Sprintf("%s: %s", label(it), it.note)
		})
		printGroup("Skipped", items, stSkip, func(it item) string {
			return fmt.Sprintf("%s: %s", label(it), it.note)
		})
	}
	printGroup("Refused before, held back (-ea retries now)", items, stWait, func(it item) string {
		return fmt.Sprintf("%s%s: %s", label(it), versionLabel(it.ver), it.note)
	})
	if eaOnly {
		fmt.Printf("\n-ea on its own: only early access versions refused before are retried")
		if notRetried > 0 {
			fmt.Printf("; %d other missing file(s) left for a normal run", notRetried)
		}
		fmt.Println(".")
	}
	if len(res.others) > 0 && !*previewsOnly && !*renameOnly {
		tags := make([]string, 0, len(res.others))
		for t := range res.others {
			tags = append(tags, t)
		}
		sort.Slice(tags, func(i, j int) bool {
			if res.others[tags[i]] != res.others[tags[j]] {
				return res.others[tags[i]] > res.others[tags[j]]
			}
			return tags[i] < tags[j]
		})
		if onCLI["base"] {
			fmt.Println("\nOther base models in your likes, not selected (add one with -base +tag):")
		} else {
			fmt.Printf("\nOther base models in your likes, not downloaded (turn one on with -reconfig, or\n\"download\": true under it in %s):\n", cfg.path)
		}
		for _, t := range tags {
			name := res.names[t]
			switch {
			case t == "":
				name = "upscalers and other models with no base model"
			case name == "":
				name = "no base model given"
			}
			fmt.Printf("  %-22s %3d   (Civitai: %s)\n", baseKey(t), res.others[t], name)
		}
	}
	if !opt.all {
		var missing []string
		for _, t := range sortedKeys(opt.bases) {
			if !res.matched[t] {
				missing = append(missing, baseKey(t))
			}
		}
		if len(missing) > 0 {
			fmt.Println("\nNo liked model uses:", strings.Join(missing, ", "))
		}
	}

	if (*local || *guess) && !eaOnly {
		var parts []string
		if *local {
			parts = append(parts, fmt.Sprintf("%d found on Civitai by checksum", ls.found))
		}
		if *guess {
			parts = append(parts, fmt.Sprintf("%d tagged from their own metadata", ls.guessed))
		}
		fmt.Printf("\nOther model files: %s.\n", strings.Join(parts, ", "))
		if len(ls.left) > 0 {
			fmt.Printf("\nLeft untagged, base model unknown (%d):\n", len(ls.left))
			for _, lf := range ls.left {
				fmt.Printf("  %s\n      %s\n", lf.rel, lf.why)
			}
			if !*guess {
				fmt.Println("  (-guess tags the ones whose own metadata names a base model)")
			}
		}
	} else if ls.notLooked > 0 && (*rename || *renameOnly) {
		fmt.Printf("\n%d other model file(s) are not liked models, so their base model is unknown; -local looks them up on Civitai by checksum.\n", ls.notLooked)
	}
	var unverified []string
	for _, l := range idx.all {
		if strings.HasSuffix(l.path, ".unverified") {
			unverified = append(unverified, idx.rel(l.path))
		}
	}
	if len(unverified) > 0 {
		fmt.Printf("\nDownloads that failed their checksum (%d), kept aside and hidden from ComfyUI:\n", len(unverified))
		for _, u := range unverified {
			fmt.Printf("  %s\n", u)
		}
		fmt.Println("  Each one of a liked model is downloaded again (see above); one that verifies replaces it.")
	}
	if odd := oddNames(idx); len(odd) > 0 {
		fmt.Printf("\nMangled names, hidden from ComfyUI (the extension is not at the end; fix these by hand):\n")
		for _, o := range odd {
			fmt.Printf("  %q\n", o)
		}
	}

	if !opt.all && !eaOnly {
		fmt.Println("\nPer base model (liked files):")
		for _, t := range sortedKeys(opt.bases) {
			c := tallies[t]
			if c == nil {
				c = &baseTally{}
			}
			if c.have+c.get+c.wait+c.alone+c.off == 0 {
				fmt.Printf("  %-16s none liked (check the tag: the full list of tags in your likes is printed by -base all -n)\n", baseKey(t))
				continue
			}
			line := fmt.Sprintf("  %-16s %d on disk, %d to download", baseKey(t), c.have, c.get)
			if *renameOnly || *previewsOnly {
				line = fmt.Sprintf("  %-16s %d on disk (%d not on disk, not downloaded in this mode)", baseKey(t), c.have, c.get)
			}
			if c.wait > 0 {
				line += fmt.Sprintf(", %d held back (refused before)", c.wait)
			}
			if c.alone > 0 {
				line += fmt.Sprintf(", %d left alone (see above)", c.alone)
			}
			if c.off > 0 {
				line += fmt.Sprintf(", %d switched off in the config", c.off)
			}
			fmt.Println(line)
		}
	}

	renames := 0
	if *rename {
		renames = untagged
		if renames > 0 {
			fmt.Printf("\nTo rename (%d):\n", renames)
			for _, it := range items {
				if it.status == stHave && it.rename != "" {
					extra := ""
					if n := len(it.sidecars); n > 0 {
						extra = fmt.Sprintf("  (+%d preview/info file(s))", n)
					}
					fmt.Printf("  %s  ->  %s%s\n", idx.rel(it.found), idx.rel(it.rename), extra)
				}
			}
		}
	} else if untagged > 0 && !*previewsOnly {
		fmt.Printf("\n%d file(s) you already have are not named the way -tag and -name say; -rename renames them in place.\n", untagged)
	}

	thumbsHave, thumbsNew, noImages := 0, 0, 0
	for _, it := range items {
		switch {
		case it.status == stHave:
			thumbsHave += len(it.thumbs)
		case it.status == stGet:
			thumbsNew += len(it.thumbs)
		default:
			continue
		}
		if *previews > 0 && len(it.images) == 0 && !previewExists(it.final(*rename), 1) {
			noImages++
		}
	}
	if thumbsHave > 0 {
		fmt.Printf("\nPreviews to fetch for files you have (%d):\n", thumbsHave)
		for _, it := range items {
			if it.status == stHave && len(it.thumbs) > 0 {
				fmt.Printf("  %s  ->  %s  (%d)\n", label(it), idx.rel(it.thumbs[0].stem)+".*", len(it.thumbs))
			}
		}
	}
	if noImages > 0 {
		fmt.Printf("\n%d model file(s) have no preview and Civitai offers no still image for them.\n", noImages)
	}
	if infoPics > 0 {
		fmt.Printf("\nGeneration settings to look up for previews you have: %d picture(s) of %d model file(s).\n", infoPics, infoFiles)
		fmt.Println("  Civitai's images for them are downloaded again to tell which picture is which.")
	} else if *previewInfo && !eaOnly {
		fmt.Println("\nEvery preview picture of a liked model you have already has its generation settings file.")
	}

	if len(copies) > 0 {
		fmt.Printf("\nThe same file stored more than once (checksums match); the first of each is kept, the rest removed:\n")
		for _, g := range copies {
			fmt.Printf("  keep    %s\n", idx.rel(g.keep.path))
			for _, d := range g.drop {
				fmt.Printf("  remove  %s  (%s)\n", idx.rel(d.path), human(d.size))
			}
		}
	}
	if len(stale) > 0 {
		fmt.Printf("\nLeftover partial or failed downloads whose finished file is already there, to remove (%d):\n", len(stale))
		for _, p := range stale {
			fmt.Printf("  %s  (%s)\n", idx.rel(p.path), human(p.size))
		}
	}
	if cleanN > 0 {
		fmt.Printf("  Frees %s.\n", human(cleanBytes))
	}

	var need int64
	for _, it := range items {
		if it.status == stGet {
			need += it.file.bytes()
		}
	}
	count := printGroup("To download", items, stGet, func(it item) string {
		kind := it.model.Type
		if it.tag != "" {
			kind += ", " + it.tag
		}
		if it.ea {
			kind += ", early access until " + eaUntil(it.ver.EarlyAccessEndsAt)
		}
		line := fmt.Sprintf("%s%s [%s]  ->  %s  (%s)", it.model.Name, versionLabel(it.ver),
			kind, idx.rel(it.dest), human(it.file.bytes()))
		if n := len(it.thumbs); n > 0 {
			line += fmt.Sprintf(" + %d preview(s)", n)
		}
		if it.note != "" {
			line += "\n      note: " + it.note
		}
		return line
	})
	if count == 0 && renames == 0 && thumbsHave == 0 && infoPics == 0 && cleanN == 0 {
		if *renameOnly {
			fmt.Printf("\nEvery model file it knows the base of is already named for -tag %s.\n", tagMode)
			return 0
		}
		if *previewsOnly {
			fmt.Printf("\nEvery model file it knows on Civitai already has its %d preview(s), or Civitai has no more to offer.\n", *previews)
			return 0
		}
		fmt.Println("\nNothing to do.")
		return 0
	}
	free, haveFree := freeBytes(root)
	if count > 0 {
		fmt.Printf("\nTotal to download: %s", human(need))
		if haveFree {
			fmt.Printf("  (free on disk: %s)", human(free))
		}
		fmt.Println()
	}
	if *dryRun {
		fmt.Println("Dry run: no models or previews changed.")
		return 0
	}
	if count > 0 && haveFree && need+spaceHeadroom > free {
		fmt.Fprintln(os.Stderr, "Not enough free disk space for all of it; nothing changed.")
		return 1
	}
	if !*yes {
		var parts []string
		if count > 0 {
			parts = append(parts, fmt.Sprintf("download %d model(s)", count))
		}
		if renames > 0 {
			parts = append(parts, fmt.Sprintf("rename %d", renames))
		}
		if t := thumbsHave + thumbsNew; t > 0 {
			parts = append(parts, fmt.Sprintf("fetch %d preview(s)", t))
		}
		if infoPics > 0 {
			parts = append(parts, fmt.Sprintf("look up settings for %d preview(s)", infoPics))
		}
		if cleanN > 0 {
			parts = append(parts, fmt.Sprintf("remove %d copie(s) and leftover(s) (%s)", cleanN, human(cleanBytes)))
		}
		question := strings.Join(parts, ", ")
		fmt.Printf("%s%s? [y/N] ", strings.ToUpper(question[:1]), question[1:])
		answer, rerr := stdin.ReadString('\n')
		if rerr != nil && strings.TrimSpace(answer) == "" {
			fmt.Println("\nNo answer to read: rerun with -yes to go ahead.")
			return 1
		}
		if a := strings.ToLower(strings.TrimSpace(answer)); a != "y" && a != "yes" {
			fmt.Println("Nothing changed.")
			return 0
		}
	}

	var failed []string
	infos := 0 // settings files written
	fetchThumbs := func(it item) int {
		saved := 0
		rel := idx.rel(it.final(*rename))
		for _, tj := range it.thumbs {
			path, err := c.fetchImage(ctx, tj.url, tj.stem)
			if err != nil {
				if ctx.Err() == nil {
					failed = append(failed, fmt.Sprintf("preview %d for %s: %v", tj.slot, label(it), err))
				}
				continue
			}
			saved++
			fmt.Printf("    preview %d  ->  %s\n", tj.slot, idx.rel(path))
			if meta := c.imageMeta(ctx, tj.img, it.model.Creator.Username); meta != nil {
				if err := writeImageInfo(path, tj.url, meta); err != nil {
					failed = append(failed, fmt.Sprintf("settings for preview %d of %s: %v", tj.slot, label(it), err))
				} else {
					infos++
				}
			}
			if e := st.Files[rel]; e != nil {
				e.Previews = append(e.Previews, tj.url)
			}
		}
		st.saveOrWarn()
		return saved
	}

	if cleanN > 0 {
		var freed int64
		removed := 0
		for _, g := range copies {
			for _, d := range g.drop {
				n, err := removeCopy(idx, st, d.path, g.keep.path)
				freed += n
				if err != nil {
					failed = append(failed, fmt.Sprintf("remove %s: %v", idx.rel(d.path), err))
					continue
				}
				removed++
			}
		}
		for _, p := range stale {
			if err := os.Remove(p.path); err != nil {
				failed = append(failed, fmt.Sprintf("remove %s: %v", idx.rel(p.path), err))
				continue
			}
			freed += p.size
			removed++
		}
		st.saveOrWarn()
		fmt.Printf("\nRemoved %d copie(s) and leftover(s), freed %s.\n", removed, human(freed))
	}

	infoMatched, infoMissing := 0, 0
	if infoPics > 0 {
		fmt.Println("\nGeneration settings for previews you have:")
		for i := range items {
			it := &items[i]
			if len(it.noInfo) == 0 {
				continue
			}
			m, miss, err := fillPreviewInfo(ctx, c, st, idx, it)
			if ctx.Err() != nil {
				fmt.Fprintln(os.Stderr, "\nInterrupted. Settings files saved so far are kept.")
				return 130
			}
			if err != nil {
				failed = append(failed, fmt.Sprintf("settings for %s: %v", label(*it), err))
			}
			infoMatched += m
			infoMissing += miss
		}
		infos += infoMatched
	}

	if renames > 0 {
		fmt.Println()
		for _, it := range items {
			if it.status != stHave || it.rename == "" {
				continue
			}
			to := it.rename
			if err := os.MkdirAll(filepath.Dir(to), 0o755); err != nil {
				failed = append(failed, fmt.Sprintf("rename %s: %v", idx.rel(it.found), err))
				continue
			}
			if _, err := os.Lstat(to); err == nil {
				failed = append(failed, fmt.Sprintf("rename %s: %s already exists", idx.rel(it.found), idx.rel(to)))
				continue
			}
			if err := os.Rename(it.found, to); err != nil {
				failed = append(failed, fmt.Sprintf("rename %s: %v", idx.rel(it.found), err))
				continue
			}
			if e := st.Files[idx.rel(it.found)]; e != nil {
				delete(st.Files, idx.rel(it.found))
				st.Files[idx.rel(to)] = e
			}
			fmt.Printf("renamed %s  ->  %s\n", idx.rel(it.found), idx.rel(to))
			cfg.note(it, "on disk", idx.rel(to))
			for _, sc := range it.sidecars {
				if _, err := os.Lstat(sc[1]); err == nil {
					continue
				}
				if err := os.Rename(sc[0], sc[1]); err != nil {
					failed = append(failed, fmt.Sprintf("rename %s: %v", idx.rel(sc[0]), err))
				}
			}
			// Previews that arrived after the plan (from a removed copy) go too.
			oldBase := strings.TrimSuffix(it.found, filepath.Ext(it.found))
			newBase := strings.TrimSuffix(to, filepath.Ext(to))
			for _, sc := range sidecars(it.found) {
				dst := newBase + strings.TrimPrefix(sc, oldBase)
				if _, err := os.Lstat(dst); errors.Is(err, os.ErrNotExist) {
					os.Rename(sc, dst)
				}
			}
			// After the previews have moved out too: a tag folder left empty goes.
			if old := filepath.Dir(it.found); isTagDir(old) && !strings.EqualFold(old, filepath.Dir(to)) {
				os.Remove(old) // fails harmlessly if anything is still in it
			}
		}
		st.saveOrWarn()
	}

	got := 0
	if thumbsHave > 0 {
		fmt.Println("\nPreviews for files you have:")
		for _, it := range items {
			if it.status == stHave && len(it.thumbs) > 0 {
				got += fetchThumbs(it)
				if ctx.Err() != nil {
					fmt.Fprintln(os.Stderr, "\nInterrupted.")
					return 130
				}
			}
		}
	}

	tty := isTerminal(os.Stderr)
	ok, done, eaNoted := 0, 0, 0
	for _, it := range items {
		if it.status != stGet {
			continue
		}
		done++
		fmt.Printf("\n[%d/%d] %s  ->  %s\n", done, count, label(it), idx.rel(it.dest))
		sum, err := c.fetchWithRetry(ctx, it, tty)
		if ctx.Err() != nil {
			fmt.Fprintln(os.Stderr, "\nInterrupted. The partial file is kept and the next run resumes it.")
			return 130
		}
		vid := strconv.FormatInt(it.ver.ID, 10)
		if err != nil {
			fmt.Fprintf(os.Stderr, "    FAILED: %v\n", err)
			if blockedByAccess(err) {
				prev := st.EarlyAccess[vid]
				rec := &eaRecord{Model: it.model.Name, Version: it.ver.Name, Tag: it.tag, Reason: err.Error(),
					BlockedAt: time.Now().UTC(), EndsAt: it.ver.EarlyAccessEndsAt,
					Availability: it.ver.Availability, UsageControl: it.ver.UsageControl}
				if rec.EndsAt == "" && prev != nil {
					rec.EndsAt = prev.EndsAt
				}
				st.EarlyAccess[vid] = rec
				eaNoted++
				cfg.note(it, "refused by Civitai: "+err.Error(), "")
				if _, ok := parseTime(rec.EndsAt); ok && it.ea {
					fmt.Fprintf(os.Stderr, "    refused: not tried again until early access ends (%s), or with -ea\n", eaUntil(rec.EndsAt))
				} else {
					fmt.Fprintln(os.Stderr, "    refused: noted, not tried again until you run with -ea")
				}
				st.saveOrWarn()
			} else {
				failed = append(failed, fmt.Sprintf("%s: %v", label(it), err))
				cfg.note(it, "download failed: "+err.Error(), "")
			}
			cfg.saveOrWarn()
			continue
		}
		delete(st.EarlyAccess, vid)
		ok++
		cfg.note(it, "on disk", idx.rel(it.dest))
		cfg.saveOrWarn()
		if it.file.sha256() != "" {
			fmt.Println("    saved, checksum verified")
		} else {
			fmt.Println("    saved (Civitai gave no checksum to verify it against)")
		}
		st.record(idx, it.dest, sum)
		st.setBase(idx, it.dest, it, res.names[it.tag])
		got += fetchThumbs(it)
	}
	st.saveOrWarn()
	cfg.saveOrWarn()
	var summary []string
	if count > 0 {
		summary = append(summary, fmt.Sprintf("Downloaded %d of %d model(s).", ok, count))
	}
	if thumbsHave+thumbsNew > 0 {
		summary = append(summary, fmt.Sprintf("Previews saved: %d.", got))
	}
	if infos > 0 || infoPics > 0 {
		summary = append(summary, fmt.Sprintf("Generation settings saved: %d.", infos))
	}
	if len(summary) > 0 {
		fmt.Println("\n" + strings.Join(summary, " "))
	}
	if infoMissing > 0 {
		fmt.Printf("%d preview picture(s) got no settings: not among Civitai's images for the model (your own\n"+
			"pictures, or changed since), or Civitai keeps no settings for them.\n", infoMissing)
	}
	if eaNoted > 0 {
		fmt.Printf("Refused by Civitai: %d, noted in the state file and held back (until early access ends, if Civitai gives a date; -ea retries).\n", eaNoted)
	}
	if len(failed) > 0 {
		fmt.Println("Failed:")
		for _, f := range failed {
			fmt.Println("  " + f)
		}
		return 1
	}
	return 0
}

// fillPreviewInfo saves settings files for the preview pictures of one model
// file that have none. Which Civitai image a picture is cannot be told from its
// name, so candidates are downloaded and compared byte for byte: the images
// recorded as saved for this file first, then the rest Civitai lists.
func fillPreviewInfo(ctx context.Context, c *client, st *stateFile, idx *index, it *item) (matched, missing int, err error) {
	const maxTries = 16
	want := map[string]string{} // sha256 -> picture still waiting for its settings
	for _, pic := range it.noInfo {
		sum, err := hashFile(ctx, pic)
		if err != nil {
			if ctx.Err() != nil {
				return matched, missing, ctx.Err()
			}
			missing++
			continue
		}
		want[sum] = pic
	}
	var order []apiImage
	seen := map[string]bool{}
	if e := st.Files[idx.rel(it.found)]; e != nil {
		for _, u := range e.Previews {
			for _, im := range it.images {
				if im.URL == u && !seen[u] {
					seen[u] = true
					order = append(order, im)
				}
			}
		}
	}
	for _, im := range it.images {
		if !seen[im.URL] {
			seen[im.URL] = true
			order = append(order, im)
		}
	}
	tries := 0
	var lastErr error
	for _, im := range order {
		if len(want) == 0 || tries >= maxTries {
			break
		}
		tries++
		data, _, err := c.imageBytes(ctx, im.URL)
		if err != nil {
			if ctx.Err() != nil {
				return matched, missing, ctx.Err()
			}
			lastErr = err
			continue
		}
		h := sha256.Sum256(data)
		pic, ok := want[hex.EncodeToString(h[:])]
		if !ok {
			continue
		}
		delete(want, hex.EncodeToString(h[:]))
		meta := c.imageMeta(ctx, im, it.model.Creator.Username)
		if meta == nil {
			missing++
			continue
		}
		if err := writeImageInfo(pic, im.URL, meta); err != nil {
			return matched, missing, err
		}
		matched++
		fmt.Printf("    %s\n", idx.rel(infoPath(pic)))
		if it.rename != "" { // the settings file follows its picture when the model is renamed
			oldBase := strings.TrimSuffix(it.found, filepath.Ext(it.found))
			newBase := strings.TrimSuffix(it.rename, filepath.Ext(it.rename))
			src := infoPath(pic)
			if dst := newBase + strings.TrimPrefix(src, oldBase); dst != src {
				it.sidecars = append(it.sidecars, [2]string{src, dst})
			}
		}
	}
	missing += len(want)
	if matched == 0 && lastErr != nil {
		return matched, missing, lastErr
	}
	return matched, missing, nil
}

func main() { os.Exit(run()) }
