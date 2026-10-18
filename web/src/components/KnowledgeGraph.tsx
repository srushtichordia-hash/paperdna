import { useCallback, useMemo, useState } from "react";
import {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  Handle,
  Position,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type {
  GraphEdgeOut,
  GraphNodeOut,
  GraphNodeData as NodeData,
} from "../api/types";

// ---- Custom node renderers, one per node kind --------------------------

const NODE_STYLES: Record<
  NodeData["kind"],
  { className: string; label: (d: NodeData) => string }
> = {
  paper: {
    className: "border-blue-300 bg-blue-50 text-blue-900",
    label: (d) =>
      d.kind === "paper" ? (d.title ?? `${d.source}:${d.sourceId}`) : "",
  },
  claim: {
    className: "border-green-300 bg-green-50 text-green-900",
    label: (d) => (d.kind === "claim" ? truncate(d.sentence, 90) : ""),
  },
  evidence: {
    className: "border-slate-300 bg-slate-50 text-slate-800",
    label: (d) => (d.kind === "evidence" ? "evidence" : ""),
  },
  condition: {
    className: "border-amber-300 bg-amber-50 text-amber-900",
    label: (d) =>
      d.kind === "condition" ? `${d.category}: ${d.value}` : "",
  },
};

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

function PaperDNANode({ data }: NodeProps) {
  const d = data as unknown as NodeData & { onSelect?: (id: string) => void };
  const style = NODE_STYLES[d.kind];
  const nodeId = (d as { id?: string }).id;
  return (
    <div
      data-testid={`graph-node-${d.kind}`}
      role="button"
      tabIndex={0}
      onClick={() => d.onSelect?.(nodeId ?? "")}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          d.onSelect?.(nodeId ?? "");
        }
      }}
      className={`cursor-pointer rounded-md border px-3 py-2 text-xs shadow-sm ${style.className}`}
      style={{ maxWidth: 220 }}
    >
      <Handle type="target" position={Position.Left} className="!bg-slate-400" />
      <div className="font-medium">{style.label(d)}</div>
      {d.kind === "evidence" && (
        <div className="mt-0.5 text-[10px] text-slate-500">
          {d.verification.replace(/_/g, " ")}
        </div>
      )}
      {d.kind === "claim" && d.start != null && (
        <div className="mt-0.5 text-[10px] text-slate-500">
          chars {d.start}–{d.end}
        </div>
      )}
      {d.kind === "condition" && d.ambiguous && (
        <div className="mt-0.5 text-[10px] text-amber-600">ambiguous</div>
      )}
      <Handle type="source" position={Position.Right} className="!bg-slate-400" />
    </div>
  );
}

const nodeTypes = { paperdna: PaperDNANode };

// ---- Graph wrapper ------------------------------------------------------

function toFlowNodes(
  nodes: GraphNodeOut[],
  onSelect: (id: string) => void
): Node[] {
  return nodes.map((n) => ({
    id: n.id,
    type: "paperdna",
    position: n.position,
    // Expose the node id and a self-selection callback to the custom node so
    // it can be selected on click/keyboard without relying on React Flow's
    // synthetic event system.
    data: {
      ...(n.data as unknown as Record<string, unknown>),
      id: n.id,
      onSelect,
    },
  }));
}

function toFlowNodesWithSelection(
  nodes: GraphNodeOut[],
  selectedId: string | null,
  onSelect: (id: string) => void
): Node[] {
  return toFlowNodes(nodes, onSelect).map((n) => ({
    ...n,
    selected: n.id === selectedId,
  }));
}

function toFlowEdges(edges: GraphEdgeOut[]): Edge[] {
  return edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    label: e.label ?? undefined,
    // Relation edges are model predictions — render them dashed to visually
    // distinguish them from observed provenance edges.
    animated: e.type === "claim_relation",
    style: {
      stroke: e.type === "claim_relation" ? "#f59e0b" : "#94a3b8",
      strokeDasharray: e.type === "claim_relation" ? "4 2" : undefined,
    },
    markerEnd: { type: MarkerType.ArrowClosed },
  }));
}

export interface KnowledgeGraphProps {
  nodes: GraphNodeOut[];
  edges: GraphEdgeOut[];
}

export default function KnowledgeGraph({ nodes, edges }: KnowledgeGraphProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const flowEdges = useMemo(() => toFlowEdges(edges), [edges]);
  const flowNodes = useMemo(
    () => toFlowNodesWithSelection(nodes, selectedId, setSelectedId),
    [nodes, selectedId]
  );

  const onNodeClick = useCallback((_evt: unknown, node: Node) => {
    setSelectedId(node.id);
  }, []);

  const selected = useMemo(
    () => nodes.find((n) => n.id === selectedId) ?? null,
    [nodes, selectedId]
  );

  if (nodes.length === 0) {
    return (
      <p className="text-sm text-slate-500">
        No graph to display (no papers or claims were found).
      </p>
    );
  }

  return (
    <div className="flex gap-3">
      <div
        data-testid="knowledge-graph"
        className="h-96 flex-1 rounded-lg border border-slate-200 bg-white"
      >
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          nodeTypes={nodeTypes}
          onNodeClick={onNodeClick}
          fitView
          proOptions={{ hideAttribution: true }}
        >
          <Background />
          <Controls />
          <MiniMap pannable zoomable />
        </ReactFlow>
      </div>
      {selected && (
        <GraphNodeDetail
          node={selected}
          onClose={() => setSelectedId(null)}
        />
      )}
    </div>
  );
}

function GraphNodeDetail({
  node,
  onClose,
}: {
  node: GraphNodeOut;
  onClose: () => void;
}) {
  const d = node.data as unknown as NodeData;
  return (
    <aside
      data-testid="graph-node-detail"
      className="w-72 rounded-lg border border-slate-200 bg-white p-4 text-sm shadow-sm"
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          {d.kind}
        </span>
        <button
          onClick={onClose}
          aria-label="Close details"
          className="text-slate-400 hover:text-slate-600"
        >
          ✕
        </button>
      </div>
      <dl className="mt-2 space-y-1">
        {Object.entries(node.data)
          .filter(([key]) => key !== "kind")
          .map(([key, value]) => (
            <div key={key}>
              <dt className="text-xs text-slate-400">{key}</dt>
              <dd className="break-words text-ink">
                {Array.isArray(value)
                  ? value.join(", ") || "—"
                  : value == null
                    ? "—"
                    : String(value)}
              </dd>
            </div>
          ))}
      </dl>
    </aside>
  );
}
