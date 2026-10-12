// civitai-likes downloads the models you have liked on Civitai into a ComfyUI
// models folder, without ever fetching one you already have.
//
// Files are saved as <base model tag>_<Civitai's file name>, for example
// flux1_dev_some_lora.safetensors, so loader lists group by base model.
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
//
// Checksums: none by default. The tool keeps a state file in the models folder
// (.civitai-likes.json) with what it has seen; each run it checksums only the
// model files that are new or changed since the last run, and uses the
// recorded checksums to spot copies saved under other names. -checksum does
// every file without one.
//
// Usage:
//
//	civitai-likes -n                 show what would happen, change nothing
//	civitai-likes                    show the plan, ask, then download
//	civitai-likes -yes               no question asked
//	civitai-likes -base +pony        the default base models plus Pony
//	civitai-likes -base sdxl1,sd1    only these base models
//	civitai-likes -base all          every base model
//	civitai-likes -rename            also rename files you already have to the tagged form
//	civitai-likes -previews 4        keep up to 4 preview images per model to choose from
//	civitai-likes -checksum          checksum every model file now (slow, once)
//	civitai-likes -latest            also fetch the newest version when you hold an older one
//
// Default base models: flux1_dev, sdxl1, qwen, sd1, krea2. Every run lists the
// other base models found in your likes with the tag to pass to -base.
//
// Build: go build -o civitai-likes civitai-likes.go   (standard library only)
package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const (
	userAgent     = "civitai-likes/1.0"
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
	ID        int64      `json:"id"`
	Name      string     `json:"name"`
	BaseModel string     `json:"baseModel"`
	Files     []apiFile  `json:"files"`
	Images    []apiImage `json:"images"`
}

type apiModel struct {
	ID            int64        `json:"id"`
	Name          string       `json:"name"`
	Type          string       `json:"type"`
	Mode          string       `json:"mode"`
	ModelVersions []apiVersion `json:"modelVersions"`
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
		return "401: Civitai rejected the API token"
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
}

// stateFile is the tool's memory between runs, kept in the models folder.
type stateFile struct {
	Version  int                   `json:"version"`
	LastScan time.Time             `json:"lastScan"`
	Files    map[string]*fileState `json:"files"` // by path relative to the models folder
	path     string
	existed  bool
}

func loadState(path string) (*stateFile, error) {
	st := &stateFile{Version: 1, Files: map[string]*fileState{}, path: path}
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
	since := st.LastScan.UnixNano()
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
			var previews []string
			if e != nil {
				previews = e.Previews
			}
			e = &fileState{Size: l.size, MTime: l.mtime, Previews: previews}
			st.Files[rel] = e
		}
		newer := l.mtime > since || l.ctime > since
		if e.SHA256 == "" && (all || (st.existed && newer)) {
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
		why := "new or changed since the last run"
		if all {
			why = "not checksummed yet"
		}
		fmt.Fprintf(os.Stderr, "Checksumming %d model file(s), %s, %s...\n", len(todo), human(todoBytes), why)
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

// --------------------------------------------------------------- previews

type apiImage struct {
	URL       string `json:"url"`
	Type      string `json:"type"`
	NSFW      any    `json:"nsfw"`
	NSFWLevel any    `json:"nsfwLevel"`
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
		for _, x := range imageExts {
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
	u, err := url.Parse(rawURL)
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") {
		return "", fmt.Errorf("unusable image link %q", rawURL)
	}
	var last error
	for attempt := 1; attempt <= 3; attempt++ {
		path, err := func() (string, error) {
			rctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
			defer cancel()
			resp, err := c.do(rctx, rawURL, map[string]string{"Accept": "image/*"})
			if err != nil {
				return "", err
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				return "", statusError(resp)
			}
			ct := strings.ToLower(strings.TrimSpace(strings.SplitN(resp.Header.Get("Content-Type"), ";", 2)[0]))
			ext, ok := imageTypes[ct]
			if !ok {
				return "", &httpError{status: http.StatusUnsupportedMediaType}
			}
			data, err := io.ReadAll(io.LimitReader(resp.Body, maxPreviewBytes+1))
			if err != nil {
				return "", err
			}
			if len(data) > maxPreviewBytes {
				return "", fmt.Errorf("image larger than %s", human(maxPreviewBytes))
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
		}()
		if err == nil {
			return path, nil
		}
		last = err
		var he *httpError
		if ctx.Err() != nil || (errors.As(err, &he) && !he.retryable()) {
			break
		}
		if attempt < 3 {
			if err := sleepCtx(ctx, backoff(attempt, err)); err != nil {
				return "", err
			}
		}
	}
	var he *httpError
	if errors.As(last, &he) && he.status == http.StatusUnsupportedMediaType {
		return "", errors.New("not an image")
	}
	return "", last
}

// -------------------------------------------------------------- base models

// The base models fetched when -base is not given.
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

// taggedName is the file name a model file gets on disk: its base model tag,
// then the name Civitai gives it.
func taggedName(tag, name string) string {
	if tag == "" || tag == "unknown" || tag == "other" {
		return name
	}
	if strings.HasPrefix(strings.ToLower(name), tag+"_") {
		return name
	}
	return tag + "_" + name
}

// parseBases reads the -base value: a comma list of tags or Civitai names.
// "all" selects everything; entries written "+tag" add to the defaults.
func parseBases(spec string) (set map[string]bool, all bool) {
	set = map[string]bool{}
	var entries []string
	extend := false
	for _, p := range strings.Split(spec, ",") {
		p = strings.TrimSpace(p)
		if strings.HasPrefix(p, "+") {
			extend = true
			p = strings.TrimSpace(p[1:])
		}
		if p != "" {
			entries = append(entries, p)
		}
	}
	if extend || len(entries) == 0 {
		for _, d := range strings.Split(defaultBases, ",") {
			set[d] = true
		}
	}
	for _, p := range entries {
		if strings.EqualFold(p, "all") {
			all = true
			continue
		}
		set[baseTag(p)] = true
	}
	return set, all
}

// -------------------------------------------------------------------- plan

const (
	stHave  = "have"  // already on disk
	stAlone = "alone" // something is in the way; left untouched
	stSkip  = "skip"  // nothing to download
	stGet   = "get"   // will be downloaded
)

type thumbJob struct {
	url  string
	stem string // path without extension
	slot int
}

type item struct {
	model    apiModel
	ver      apiVersion
	file     apiFile
	tag      string // base model tag, "" for types without one
	status   string
	note     string
	dest     string      // where a download goes (stGet)
	found    string      // where the file already is (stHave)
	rename   string      // the name `found` should have to carry its tag; "" if it does
	images   []apiImage  // preview candidates, best first
	thumbs   []thumbJob  // previews to fetch
	sidecars [][2]string // files that move along with a rename
}

// final is where the model file is, or will be, once this run is done.
func (it item) final(renaming bool) string {
	switch {
	case it.status == stGet:
		return it.dest
	case renaming && it.rename != "":
		return filepath.Join(filepath.Dir(it.found), it.rename)
	}
	return it.found
}

type candidate struct {
	ver    apiVersion
	file   apiFile
	tag    string
	tagged string // file name on disk: base tag + Civitai's name
}

type planOptions struct {
	latest   bool
	bases    map[string]bool
	all      bool
	rename   bool
	previews int
}

type planResult struct {
	items   []item
	others  map[string]int    // base tag not selected -> liked models that have it
	names   map[string]string // base tag -> what Civitai calls it
	matched map[string]bool   // selected tags that at least one liked model has
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

	// evaluate decides what to do about one model for one base model.
	// cands are that base's versions, newest first.
	evaluate := func(m apiModel, dir string, cands []candidate) item {
		target := cands[0]
		it := item{model: m, ver: target.ver, file: target.file, tag: target.tag}
		it.images = previewImages(target.ver, cands)
		check := cands
		if opt.latest {
			check = cands[:1]
		}
		have := func(l localFile, cd candidate, how string, i int) item {
			it.status, it.found = stHave, l.path
			it.images = previewImages(cd.ver, cands)
			it.note = idx.rel(l.path) + how
			if i > 0 {
				it.note += fmt.Sprintf("; that is version %q, the newest is %q", cd.ver.Name, target.ver.Name)
			}
			if !strings.EqualFold(filepath.Base(l.path), cd.tagged) {
				it.rename = cd.tagged
			}
			return it
		}

		// 1. On disk under its tagged name or under Civitai's own name, anywhere
		//    in the tree, with the size agreeing.
		for i, cd := range check {
			for _, name := range []string{cd.tagged, cd.file.Name} {
				for _, l := range idx.byName[strings.ToLower(name)] {
					if cd.file.bytes() == 0 || sizeClose(l.size, cd.file.bytes()) {
						return have(l, cd, "", i)
					}
				}
			}
		}
		// 2. The name it would be saved under belongs to a different file.
		if ls := idx.byName[strings.ToLower(target.tagged)]; len(ls) > 0 {
			it.status = stAlone
			it.note = fmt.Sprintf("%s already has the name this would be saved under, with a different size (yours %s, Civitai's %s)",
				idx.rel(ls[0].path), human(ls[0].size), human(target.file.bytes()))
			return it
		}
		// 3. An earlier download of it that failed its checksum.
		if ls := idx.byName[strings.ToLower(target.tagged)+".unverified"]; len(ls) > 0 {
			it.status = stAlone
			it.note = idx.rel(ls[0].path) + " is an earlier download that failed its checksum; delete it to retry"
			return it
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
					it.note = fmt.Sprintf("%s is exactly the same size and has not been checksummed; run with -checksum to settle it", idx.rel(l.path))
				}
				return it
			}
		}
		// 6. Missing: download the newest version for this base model.
		key := strings.ToLower(target.tagged)
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
			it.dest = filepath.Join(idx.root, dir, target.tagged)
			if ls := idx.byName[strings.ToLower(target.file.Name)]; len(ls) > 0 && target.tagged != target.file.Name {
				it.note = fmt.Sprintf("%s has Civitai's name for it but a different size, so it is not this file", idx.rel(ls[0].path))
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
			groups[tag] = append(groups[tag], candidate{v, f, tag, taggedName(tag, f.Name)})
		}
		if len(order) == 0 {
			note := "no downloadable file"
			if m.Mode != "" {
				note += " (" + strings.ToLower(m.Mode) + " on Civitai)"
			}
			res.items = append(res.items, item{model: m, status: stSkip, note: note})
			continue
		}
		for _, tag := range order {
			if !agnostic {
				if !opt.all && !opt.bases[tag] {
					res.others[tag]++
					continue
				}
				res.matched[tag] = true
			}
			res.items = append(res.items, evaluate(m, dir, groups[tag]))
		}
	}

	// A file is renamed only when nothing has the new name and no other liked
	// model lays claim to the same file or the same new name. Its previews and
	// info files go with it.
	srcTaken, dstTaken := map[string]bool{}, map[string]bool{}
	for i := range res.items {
		it := &res.items[i]
		if it.status != stHave || it.rename == "" {
			continue
		}
		to := filepath.Join(filepath.Dir(it.found), it.rename)
		_, err := os.Lstat(to)
		if err == nil || srcTaken[it.found] || dstTaken[strings.ToLower(to)] {
			it.note += fmt.Sprintf("; cannot be renamed, %s is taken", it.rename)
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
	if opt.previews > 0 {
		for i := range res.items {
			it := &res.items[i]
			if it.status != stHave && it.status != stGet {
				continue
			}
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
			for slot := 1; slot <= opt.previews; slot++ {
				if previewExists(current, slot) {
					continue
				}
				for next < len(it.images) && used[it.images[next].URL] {
					next++
				}
				if next >= len(it.images) {
					break
				}
				it.thumbs = append(it.thumbs, thumbJob{url: it.images[next].URL, stem: previewStem(final, slot), slot: slot})
				used[it.images[next].URL] = true
				next++
			}
		}
	}
	return res
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
			return "", fmt.Errorf("%w: kept as %s, not under its real name", errChecksum, filepath.Base(bad))
		}
	}
	if _, err := os.Lstat(it.dest); err == nil {
		return "", fmt.Errorf("%s appeared while downloading; the new copy stays as %s", it.dest, filepath.Base(part))
	}
	return got, os.Rename(part, it.dest)
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

func run() int {
	home, _ := os.UserHomeDir()
	modelsDir := flag.String("models", filepath.Join(home, "ComfyUI", "models"), "ComfyUI models folder")
	dryRun := flag.Bool("n", false, "show what would happen; change nothing")
	yes := flag.Bool("yes", false, "go ahead without asking")
	baseSpec := flag.String("base", defaultBases, "base models to fetch, comma separated, as tags (flux1_dev) or\n"+
		"Civitai names (\"Flux.1 D\"); \"+pony,+illustrious\" adds to the defaults; \"all\" takes every base")
	rename := flag.Bool("rename", false, "rename files you already have so their names start with the base model tag")
	previews := flag.Int("previews", 1, "preview images to keep per model file; the first is the one ComfyUI shows,\n"+
		"extras are saved as <name>.preview-2.jpg and so on for you to choose from (0 = none)")
	previewsOnly := flag.Bool("previews-only", false, "only fetch missing previews for liked models you already have, up to\n"+
		"-previews per file; no model downloads or renames")
	checksum := flag.Bool("checksum", false, "checksum every model file that has no checksum on record (slow; normally\n"+
		"only files new or changed since the last run are checksummed)")
	latest := flag.Bool("latest", false, "fetch the newest version even when an older version is on disk")
	maxModels := flag.Int("max", 1000, "refuse to act on a likes list longer than this")
	stateOpt := flag.String("state", "", "state file (default <models>/.civitai-likes.json)")
	tokenFile := flag.String("token-file", filepath.Join(home, ".config", "civitai", "token"), "file holding the API token")
	apiBase := flag.String("api", "https://civitai.com", "Civitai base URL")
	flag.Parse()
	if flag.NArg() > 0 {
		fmt.Fprintln(os.Stderr, "unexpected argument:", flag.Arg(0))
		return 2
	}
	if *previews < 0 {
		*previews = 0
	}
	if *previewsOnly {
		if *previews == 0 {
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
	statePath := *stateOpt
	if statePath == "" {
		statePath = filepath.Join(root, ".civitai-likes.json")
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

	opt := planOptions{latest: *latest, rename: *rename, previews: *previews}
	opt.bases, opt.all = parseBases(*baseSpec)
	fmt.Printf("Liked models: %d\n", len(models))
	fmt.Printf("Files under %s: %d (%s)\n", root, len(idx.all), human(idx.bytes))
	if opt.all {
		fmt.Println("Base models: all")
	} else {
		fmt.Println("Base models:", strings.Join(sortedKeys(opt.bases), ", "))
	}
	if !st.existed {
		fmt.Println("First run: file sizes and dates recorded. From the next run on, model files that are new")
		fmt.Println("or changed are checksummed automatically; -checksum does every file now (slow).")
	}

	res := plan(c, idx, st, models, opt)
	items := res.items
	notOnDisk := 0
	if *previewsOnly {
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
	if *previewsOnly {
		haveN := 0
		for _, it := range items {
			if it.status == stHave {
				haveN++
			}
		}
		fmt.Printf("\nPreviews only, up to %d per file. Liked model files on disk: %d", *previews, haveN)
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
	if len(res.others) > 0 && !*previewsOnly {
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
		fmt.Println("\nOther base models in your likes, not selected (add one with -base +tag):")
		for _, t := range tags {
			name := res.names[t]
			if name == "" {
				name = "no base model given"
			}
			fmt.Printf("  %-22s %3d   (Civitai: %s)\n", t, res.others[t], name)
		}
	}
	if !opt.all {
		var missing []string
		for _, t := range sortedKeys(opt.bases) {
			if !res.matched[t] {
				missing = append(missing, t)
			}
		}
		if len(missing) > 0 {
			fmt.Println("\nNo liked model uses:", strings.Join(missing, ", "))
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
					fmt.Printf("  %s  ->  %s%s\n", idx.rel(it.found), it.rename, extra)
				}
			}
		}
	} else if untagged > 0 && !*previewsOnly {
		fmt.Printf("\n%d file(s) you already have do not carry their base model in the name; -rename renames them in place.\n", untagged)
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
	if count == 0 && renames == 0 && thumbsHave == 0 {
		if *previewsOnly {
			fmt.Printf("\nEvery liked model file on disk already has its %d preview(s), or Civitai has no more to offer.\n", *previews)
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
		question := strings.Join(parts, ", ")
		fmt.Printf("%s%s? [y/N] ", strings.ToUpper(question[:1]), question[1:])
		answer, rerr := bufio.NewReader(os.Stdin).ReadString('\n')
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
			if e := st.Files[rel]; e != nil {
				e.Previews = append(e.Previews, tj.url)
			}
		}
		st.saveOrWarn()
		return saved
	}

	if renames > 0 {
		fmt.Println()
		for _, it := range items {
			if it.status != stHave || it.rename == "" {
				continue
			}
			to := filepath.Join(filepath.Dir(it.found), it.rename)
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
			fmt.Printf("renamed %s  ->  %s\n", idx.rel(it.found), it.rename)
			for _, sc := range it.sidecars {
				if _, err := os.Lstat(sc[1]); err == nil {
					continue
				}
				if err := os.Rename(sc[0], sc[1]); err != nil {
					failed = append(failed, fmt.Sprintf("rename %s: %v", idx.rel(sc[0]), err))
				}
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
	ok, done := 0, 0
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
		if err != nil {
			fmt.Fprintf(os.Stderr, "    FAILED: %v\n", err)
			failed = append(failed, fmt.Sprintf("%s: %v", label(it), err))
			continue
		}
		ok++
		if it.file.sha256() != "" {
			fmt.Println("    saved, checksum verified")
		} else {
			fmt.Println("    saved (Civitai gave no checksum to verify it against)")
		}
		st.record(idx, it.dest, sum)
		got += fetchThumbs(it)
	}
	st.saveOrWarn()
	if count > 0 {
		fmt.Printf("\nDownloaded %d of %d model(s).", ok, count)
	}
	if thumbsHave+thumbsNew > 0 {
		fmt.Printf(" Previews saved: %d.", got)
	}
	fmt.Println()
	if len(failed) > 0 {
		fmt.Println("Failed:")
		for _, f := range failed {
			fmt.Println("  " + f)
		}
		return 1
	}
	return 0
}

func main() { os.Exit(run()) }
