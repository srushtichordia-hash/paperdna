import "@testing-library/jest-dom/vitest";

// jsdom does not implement ResizeObserver, which @xyflow/react (React Flow)
// uses to measure its container. Provide a minimal stub so the graph renders
// in tests without crashing.
if (typeof globalThis.ResizeObserver === "undefined") {
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  globalThis.ResizeObserver =
    ResizeObserverStub as unknown as typeof ResizeObserver;
}
