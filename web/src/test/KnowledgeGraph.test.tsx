import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import KnowledgeGraph from "../components/KnowledgeGraph";
import { EmptyState, ErrorState, LoadingState } from "../components/StateStates";
import type { GraphNodeOut } from "../api/types";

describe("State surfaces", () => {
  it("renders a loading state with a status role", () => {
    render(<LoadingState />);
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByTestId("spinner")).toBeInTheDocument();
  });

  it("renders an empty state message", () => {
    render(<EmptyState message="Nothing found." />);
    expect(screen.getByText("Nothing found.")).toBeInTheDocument();
  });

  it("renders an error state with an alert role and retry", async () => {
    const onRetry = vi.fn();
    const user = userEvent.setup();
    render(<ErrorState message="Boom." onRetry={onRetry} />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /try again/i }));
    expect(onRetry).toHaveBeenCalled();
  });
});

describe("KnowledgeGraph", () => {
  const nodes: GraphNodeOut[] = [
    {
      id: "demo:p001",
      type: "paper",
      position: { x: 0, y: 0 },
      data: {
        kind: "paper",
        source: "demo",
        sourceId: "p001",
        title: "A Paper",
        url: null,
        doi: null,
        published: null,
        authors: [],
      },
    },
    {
      id: "demo:p001:0-10",
      type: "claim",
      position: { x: 300, y: 0 },
      data: {
        kind: "claim",
        source: "demo",
        sourceId: "p001",
        sentence: "Retrieval helps.",
        start: 0,
        end: 10,
        cue: null,
        hedges: [],
      },
    },
  ];

  it("renders a node for each graph node", () => {
    render(<KnowledgeGraph nodes={nodes} edges={[]} />);
    expect(screen.getByTestId("graph-node-paper")).toBeInTheDocument();
    expect(screen.getByTestId("graph-node-claim")).toBeInTheDocument();
  });

  it("shows an empty message when there are no nodes", () => {
    render(<KnowledgeGraph nodes={[]} edges={[]} />);
    expect(screen.getByText(/no graph to display/i)).toBeInTheDocument();
  });

  it("opens the detail panel on node click", () => {
    // React Flow measures the container; jsdom reports zero size, so stub it.
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
      width: 800,
      height: 400,
      top: 0,
      left: 0,
      right: 800,
      bottom: 400,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect);

    // fireEvent.click triggers the node's React onClick without the mousedown
    // that React Flow's d3-drag layer attaches, which would throw in jsdom
    // (d3-drag reads event.document, unavailable outside a real browser).
    render(<KnowledgeGraph nodes={nodes} edges={[]} />);
    fireEvent.click(screen.getByTestId("graph-node-paper"));
    expect(screen.getByTestId("graph-node-detail")).toBeInTheDocument();
  });
});
