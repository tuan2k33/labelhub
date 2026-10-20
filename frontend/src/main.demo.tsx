import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { installDemo } from "./demo/mock";
import "./styles.css";

installDemo();
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
