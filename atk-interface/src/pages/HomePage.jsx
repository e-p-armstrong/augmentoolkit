import React, { useEffect } from 'react';
import { useNavigation } from '../context/NavigationContext';

function HomePage() {
  const { setHighlightedNavLink } = useNavigation();

  useEffect(() => {
    setHighlightedNavLink('/configs'); // highlight configs when mounted
    return () => setHighlightedNavLink(null); // clear on unmount
  }, [setHighlightedNavLink]);

  return (
    <div className="p-8 flex flex-col items-center justify-center h-full">
      <h1 className="text-4xl font-bold mb-4">Welcome to Augmentoolkit</h1>
      <p className="text-xl text-gray-400">Select a section from the navigation above to get started.</p>
      {/* Placeholder for logo or other content */}
    </div>
  );
}

export default HomePage; 