import React from 'react';
import { FiX } from 'react-icons/fi';

function Modal({ isOpen, onClose, title, children, width = 'max-w-lg' }) {

  // Handle Escape key press - Hooks must be called unconditionally
  React.useEffect(() => {
    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    // Only add listener if modal is open
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
    }
    // Cleanup function removes listener
    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen, onClose]); // Depend on isOpen and onClose

  // Early return if not open, *after* hooks
  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-70 z-50 flex items-center justify-center p-4 transition-opacity duration-300 ease-in-out opacity-100" // Start visible for animation
      onClick={onClose} // Close on backdrop click
      aria-modal="true"
      role="dialog"
    >
      <div 
        className={`bg-gray-800 rounded-lg shadow-xl p-6 w-full ${width} relative transform transition-all duration-300 ease-in-out scale-95 opacity-0 animate-fade-scale-in`} 
        onClick={e => e.stopPropagation()} // Prevent closing when clicking inside modal
      >
        <button
          onClick={onClose}
          className="absolute top-3 right-3 text-gray-500 hover:text-gray-300 text-2xl p-1 rounded-full hover:bg-gray-700"
          aria-label="Close modal"
        >
          <FiX />
        </button>
        <h2 className="text-xl font-semibold mb-4 text-gray-100">{title}</h2>
        <div className="text-gray-300">
            {children}
        </div>
      </div>
    </div>
  );
}

// Add simple fade-in animation (requires Tailwind config update)
// In tailwind.config.js -> theme -> extend -> keyframes & animation:
/*
keyframes: {
  'fade-scale-in': {
    '0%': { opacity: '0', transform: 'scale(0.95)' },
    '100%': { opacity: '1', transform: 'scale(1)' },
  }
},
animation: {
  'fade-scale-in': 'fade-scale-in 0.2s ease-out forwards',
}
*/

export default Modal; 