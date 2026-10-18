import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import SearchForm from "../components/SearchForm";

describe("SearchForm", () => {
  it("renders the claim input and analyze button", () => {
    render(<SearchForm onSubmit={() => {}} />);
    expect(screen.getByTestId("claim-input")).toBeInTheDocument();
    expect(screen.getByTestId("analyze-button")).toHaveTextContent(
      "Analyze Research"
    );
  });

  it("submits the claim on submit", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<SearchForm onSubmit={onSubmit} />);

    await user.type(screen.getByTestId("claim-input"), "A test claim.");
    await user.click(screen.getByTestId("analyze-button"));

    expect(onSubmit).toHaveBeenCalledWith({
      claim: "A test claim.",
      abstract: null,
    });
  });

  it("shows an error and does not submit when claim is empty", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<SearchForm onSubmit={onSubmit} />);

    await user.click(screen.getByTestId("analyze-button"));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/enter a scientific claim/i);
  });

  it("reveals the abstract input in abstract mode and submits it", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<SearchForm onSubmit={onSubmit} />);

    await user.click(screen.getByRole("tab", { name: /claim \+ abstract/i }));
    await user.type(screen.getByTestId("claim-input"), "A claim.");
    await user.type(screen.getByTestId("abstract-input"), "Some abstract.");
    await user.click(screen.getByTestId("analyze-button"));

    expect(onSubmit).toHaveBeenCalledWith({
      claim: "A claim.",
      abstract: "Some abstract.",
    });
  });
});
