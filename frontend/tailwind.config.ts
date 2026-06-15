import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#182026",
        mist: "#f5f7f4",
        moss: "#46614f",
        coral: "#c95f4d",
        steel: "#536d7a"
      }
    }
  },
  plugins: []
};

export default config;

