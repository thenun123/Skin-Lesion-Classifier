/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        clinical: {
          teal: '#0F766E',
          tealLight: '#14B8A6',
          sky: '#0EA5E9',
          bg: '#F8FAFC',
          surface: '#FFFFFF',
          text: '#0F172A',
          textMuted: '#475569',
          border: '#E2E8F0',
          benign: '#16A34A',
          benignBg: '#F0FDF4',
          malignant: '#DC2626',
          malignantBg: '#FEF2F2',
        },
      },
      fontFamily: {
        heading: ['Manrope', 'sans-serif'],
        body: ['Inter', 'sans-serif'],
      },
      boxShadow: {
        soft: '0 4px 24px -8px rgba(15, 118, 110, 0.12)',
        softLg: '0 12px 40px -12px rgba(15, 118, 110, 0.18)',
      },
    },
  },
  plugins: [],
}
