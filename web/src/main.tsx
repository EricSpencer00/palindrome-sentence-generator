import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import App from "./App.tsx"

/* Three routes, one page.
 *
 *   /      v1 — a search writes a palindrome and you watch it happen
 *   /dev   v3 — composed from verified mirror-pairs, written out the same way
 *   /v2    v2 — the same poster, placing whole attested sentences
 *
 * `App` covers all three: they arrive as the same Shape, so the grid, the
 * camera and clicking a word to find its mirror do not care which one they
 * are drawing. Which route is which is decided in App.tsx. */
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
