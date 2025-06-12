import React, { createContext, useContext, useState } from 'react';

// Context to store which nav link should be highlighted (in addition to the active one)
const NavigationContext = createContext({
  highlightedNavLink: null,
  setHighlightedNavLink: () => {},
});

export const NavigationProvider = ({ children }) => {
  const [highlightedNavLink, setHighlightedNavLink] = useState(null);

  return (
    <NavigationContext.Provider value={{ highlightedNavLink, setHighlightedNavLink }}>
      {children}
    </NavigationContext.Provider>
  );
};

export const useNavigation = () => useContext(NavigationContext); 