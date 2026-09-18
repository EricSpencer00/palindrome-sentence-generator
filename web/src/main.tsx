import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import App from "./App.tsx"

/* Every prior public route now renders the same withdrawal notice. The old
 * generator and its historical outputs must not remain reachable through a
 * static client while the evidence-gated method is under construction. */
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
