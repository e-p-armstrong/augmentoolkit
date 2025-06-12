/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Add custom colors if needed
      },
      keyframes: {
        'fade-scale-in': {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        shake: {
          '0%, 100%': { transform: 'translateX(0)' },
          '10%, 30%, 50%, 70%, 90%': { transform: 'translateX(-2px)' },
          '20%, 40%, 60%, 80%': { transform: 'translateX(2px)' },
        },
        'pulse-red': {
          '0%, 100%': { backgroundColor: '#dc2626', color: 'white' }, // bg-red-600 text-white
          '50%': { backgroundColor: '#374151', color: '#f87171' }, // bg-gray-700 text-red-400
        },
        'pulse-red-intense': { // For Clear All Logs button
          '0%, 100%': { backgroundColor: '#991b1b', color: '#fecaca' }, // bg-red-800 text-red-200
          '50%': { backgroundColor: '#ef4444', color: 'white' },    // bg-red-500 text-white
        },
        'pulse-orange': { // For Delete Selected button
          '0%, 100%': { backgroundColor: '#374151', color: '#fb923c' }, // bg-gray-700 text-orange-400
          '50%': { backgroundColor: '#f97316', color: 'white' },    // bg-orange-500 text-white
        },
        'pulse-green-download': { // New animation for download button
          '0%, 100%': { backgroundColor: '#374151', color: '#34d399' }, // bg-gray-700 text-green-400
          '50%': { backgroundColor: '#16a34a', color: 'white' }, // bg-green-600 text-white
        },
        'pulse-bg-subtle': {
          '0%, 100%': { backgroundColor: 'rgba(55, 65, 81, 0.5)' }, // bg-gray-700 opacity-80 (Start)
          '50%': { backgroundColor: 'rgba(209, 213, 219, 0.4)' }, // bg-gray-300 opacity-70 (Pulse target - lighter)
        },
        'pulse-border-subtle': {
          '0%, 100%': { borderColor: 'rgba(75, 85, 99, 1)' }, // border-gray-600 (Start)
          '50%': { borderColor: 'rgba(229, 231, 235, 1)' }, // border-gray-200 (Pulse target - lighter)
        },
        'pulse-blue-gold': {
          '0%, 100%': { backgroundColor: '#2563eb', color: 'white' }, // bg-blue-600
          '50%': { backgroundColor: '#f59e0b', color: '#1f2937' }, // bg-amber-500 text-gray-800
        },
        'pulse-white': {
          '0%, 100%': { backgroundColor: 'rgba(255, 255, 255, 0.1)' },
          '50%': { backgroundColor: 'rgba(255, 255, 255, 0.3)' },
        },
        'pulse-text-subtle': { // New keyframe for text color
          '0%, 100%': { color: '#9ca3af' }, // text-gray-400 (was -300)
          '50%': { color: '#ffffff' },    // white (was text-gray-50)
        },
        'pulse-text-mild': {
          '0%, 100%': { color: '#9ca3af' }, // text-gray-400 (was -300)
          '50%': { color: '#ffffff' },    // white (was text-gray-50)
        }
      },
      animation: {
        'fade-scale-in': 'fade-scale-in 0.2s ease-out forwards',
        shake: 'shake 0.4s ease-in-out',
        'pulse-red': 'pulse-red 1.2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-red-intense': 'pulse-red-intense 1.0s cubic-bezier(0.4, 0, 0.6, 1) infinite', // Faster pulse
        'pulse-orange': 'pulse-orange 1.2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-green-download': 'pulse-green-download 1.2s cubic-bezier(0.4, 0, 0.6, 1) infinite', // Matched duration to pulse-red
        'pulse-bg-subtle': 'pulse-bg-subtle 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-border-subtle': 'pulse-border-subtle 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-blue-gold': 'pulse-blue-gold 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-white': 'pulse-white 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-text-subtle': 'pulse-text-subtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite', // New animation class
      }
    },
  },
  plugins: [],
}

