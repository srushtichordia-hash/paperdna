// Reusable UI state surfaces: loading, empty, and error. Kept small and
// presentational so they are trivially unit-testable.

export function LoadingState({ message }: { message?: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white p-6 text-slate-600"
    >
      <span
        data-testid="spinner"
        className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-accent"
      />
      <span>{message ?? "Analyzing…"}</span>
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div
      role="status"
      className="rounded-lg border border-slate-200 bg-white p-6 text-slate-600"
    >
      {message}
    </div>
  );
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div
      role="alert"
      className="rounded-lg border border-red-200 bg-red-50 p-6 text-red-800"
    >
      <p className="font-medium">Something went wrong</p>
      <p className="mt-1 text-sm">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-3 rounded-md border border-red-300 px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-100"
        >
          Try again
        </button>
      )}
    </div>
  );
}
