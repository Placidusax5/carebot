/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eefbf5",
          100: "#d6f5e6",
          200: "#aeeacf",
          300: "#7adab3",
          400: "#46c093",
          500: "#22a377",
          600: "#16825f",
          700: "#14684d",
          800: "#14533f",
          900: "#124535",
        },
      },
    },
  },
  plugins: [],
};
