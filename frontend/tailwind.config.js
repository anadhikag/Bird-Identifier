/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        forest: {
          50: '#f4f7f5',
          100: '#e4ebe6',
          200: '#cbdad0',
          300: '#a3beae',
          400: '#759b83',
          500: '#537d63',
          600: '#3f624d',
          700: '#344f3e',
          800: '#2b3f33',
          900: '#1b2620',
          950: '#0f1713', // Deep Forest Green
        },
        moss: {
          50: '#f5f7f2',
          100: '#e7ede0',
          200: '#d1dbc6',
          300: '#b0c2a2',
          400: '#8ca67a',
          500: '#6c8a5a', // Moss Green
          600: '#536d44',
          700: '#415535',
          800: '#36452d',
          900: '#2f3b28',
        },
        bark: {
          50: '#faf7f5',
          100: '#f3ece7',
          200: '#e7d8ce',
          300: '#d2baa7',
          400: '#b7967c',
          500: '#9f785c',
          600: '#906449',
          700: '#78503b',
          800: '#634333',
          900: '#2c1e17', // Dark Bark Brown
          950: '#1a110d',
        },
        ivory: {
          50: '#fefefe',
          100: '#fcfbf7', // Warm Ivory
          200: '#f6f3e6',
          300: '#eae3cc',
          400: '#dacfaa',
          500: '#c5b485',
          600: '#b09d66',
          700: '#94814d',
          800: '#79683d',
          900: '#635432',
        },
        gold: {
          100: '#f8f4eb',
          200: '#ecdcb9',
          300: '#dfc288',
          400: '#d2a85d',
          500: '#bfa063', // Muted Gold
          600: '#997e4b',
          700: '#735c36',
          800: '#4d3d24',
          900: '#281f12',
        },
        slate: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b', // Slate
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
        }
      },
      fontFamily: {
        serif: ['"Playfair Display"', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 1s ease-out forwards',
        'fade-in-slow': 'fadeIn 1.6s ease-out forwards',
        'slide-up': 'slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'float-slow': 'floatSlow 6s ease-in-out infinite',
        'sway': 'sway 8s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(30px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        floatSlow: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        sway: {
          '0%, 100%': { transform: 'rotate(-1deg) translateY(0px)' },
          '50%': { transform: 'rotate(1deg) translateY(-4px)' },
        }
      }
    },
  },
  plugins: [],
}
