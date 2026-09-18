import { render, screen } from "@testing-library/react"
import { expect, test } from "vitest"

import App from "./App"


test("public surface gives a clear unavailable status without a generator control", () => {
  render(<App />)

  expect(screen.getByRole("heading", { name: /generation is unavailable/i })).toBeTruthy()
  expect(screen.getByText(/blinded human reader evidence/i)).toBeTruthy()
  expect(screen.queryByRole("button")).toBeNull()
})
