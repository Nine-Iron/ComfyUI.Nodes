import { app } from "../../scripts/app.js";

const NODE_TYPE = "TaggedImageSaver";

function stem(value) {
  return String(value).split(/[\\/]/).pop().replace(/\.json$/i, "").trim();
}

function titleWorkflowName() {
  return document.title
    .replace(/^(\[[^\]]*\]\s*)+/, "")
    .replace(/^\*\s*/, "")
    .replace(/\s+-\s+ComfyUI.*$/i, "");
}

const NAME_SOURCES = [
  () => app.extensionManager?.workflow?.activeWorkflow?.filename,
  () => app.extensionManager?.workflow?.activeWorkflow?.path,
  () => app.workflowManager?.activeWorkflow?.name,
  titleWorkflowName,
];

export function activeWorkflowName() {
  for (const source of NAME_SOURCES) {
    try {
      const value = source();
      if (value && stem(value)) return stem(value);
    } catch {
      continue;
    }
  }
  return "unsaved";
}

function resolveWorkflowName(widget) {
  const typed = typeof widget.value === "string" ? widget.value.trim() : "";
  return !typed || typed.toLowerCase() === "auto" ? activeWorkflowName() : typed;
}

app.registerExtension({
  name: "TaggedImageSaver.WorkflowName",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);

      const nameWidget = this.widgets?.find((w) => w.name === "workflow_name");
      if (nameWidget) nameWidget.serializeValue = () => resolveWorkflowName(nameWidget);

      const lastSaved = this.addWidget("text", "last_saved", "", () => {}, { serialize: false });
      lastSaved.serialize = false;
      this.taggedLastSaved = lastSaved;

      return result;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const saved = message?.text?.[0];
      if (saved && this.taggedLastSaved) {
        this.taggedLastSaved.value = saved;
        this.setDirtyCanvas?.(true, true);
      }
    };
  },
});
