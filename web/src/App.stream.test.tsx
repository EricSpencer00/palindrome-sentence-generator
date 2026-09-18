import { render, screen } from "@testing-library/react"
import { expect, test } from "vitest"

import App from "./App"


test("every route surface has no request or stream control", () => {
  render(<App />)

  expect(screen.queryByLabelText(/prompt/i)).toBeNull()
  expect(screen.queryByRole("button", { name: /generate/i })).toBeNull()
})
