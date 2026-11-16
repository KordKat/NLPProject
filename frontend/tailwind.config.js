import typography from "@tailwindcss/typography";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#050a16",
          900: "#081022",
          850: "#0b152b",
          800: "#0f1b35",
          700: "#16264a",
          600: "#1f335f",
          500: "#2c4479",
        },
        accent: {
          blue: "#3b82f6",
          sky: "#38bdf8",
          teal: "#14b8a6",
          mint: "#5eead4",
        },
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "Sarabun", "Noto Sans Thai", "system-ui", "sans-serif"],
        serif: ["Source Serif 4", "Georgia", "serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        panel: "0 12px 40px -12px rgba(0,0,0,0.55)",
        glow: "0 0 0 1px rgba(56,189,248,0.25), 0 8px 30px -8px rgba(20,184,166,0.35)",
      },
    },
  },
  plugins: [typography],
};
