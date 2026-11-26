import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Shows what Image Recall will load as soon as an image is picked or dropped,
// before anything is queued.

const NODE_TYPE = "ImageRecall";

app.registerExtension({
  name: "ImageRecall.Info",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const node = this;

      const box = ComfyWidgets.STRING(node, "recalled", ["STRING", { multiline: true }], app).widget;
      box.serialize = false;
      box.options = { ...(box.options || {}), serialize: false };
      if (box.inputEl) {
        box.inputEl.readOnly = true;
        box.inputEl.style.opacity = "0.85";
        box.inputEl.placeholder = "Drop an image on this node to see what it was made with.";
      }
      box.value = "";

      const widget = (name) => node.widgets?.find((w) => w.name === name);
      let pending = 0;
      node.imageRecallRefresh = async () => {
        const image = widget("image")?.value;
        if (!image) return;
        const ticket = ++pending;
        const q = new URLSearchParams({
          filename: String(image),
          checkpoint: String(widget("checkpoint")?.value ?? "(from image)"),
          apply_loras: String(widget("apply_loras")?.value ?? true),
        });
        let text;
        try {
          const res = await api.fetchApi(`/image_recall/info?${q}`);
          text = (await res.json()).text;
        } catch (err) {
          text = `Could not read metadata: ${err}`;
        }
        if (ticket !== pending) return; // a newer image was picked meanwhile
        box.value = text ?? "";
        node.setDirtyCanvas?.(true, true);
      };

      for (const name of ["image", "checkpoint", "apply_loras"]) {
        const w = widget(name);
        if (!w) continue;
        const cb = w.callback;
        w.callback = function () {
          const r = cb?.apply(this, arguments);
          node.imageRecallRefresh();
          return r;
        };
      }
      setTimeout(() => node.imageRecallRefresh(), 0);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => this.imageRecallRefresh?.(), 0);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function () {
      onExecuted?.apply(this, arguments);
      this.imageRecallRefresh?.();
    };
  },
});
