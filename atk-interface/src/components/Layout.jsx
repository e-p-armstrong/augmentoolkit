import React, { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useNavigation } from '../context/NavigationContext';
import atkLogo from '../assets/atk-logo-notscuffed.png';

const animationStyles = `
  @keyframes pulse-medium-white {
    0%, 100% { background-color: rgba(255,255,255,0.1); }
    50% { background-color: rgba(255,255,255,0.3); }
  }
  .animate-pulse-medium-white {
    animation: pulse-medium-white 1.8s infinite;
  }
`;

function Layout() {
  const { highlightedNavLink, setHighlightedNavLink } = useNavigation();
  const location = useLocation();
  const [serverOnline, setServerOnline] = useState(false);
  const [datagenServerOnline, setDatagenServerOnline] = useState(false);

  // Check for server on port 8003
  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8003/health', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        });
        
        const isOnline = response.ok;
        setServerOnline(isOnline);
        
        // Auto-highlight chat button when server comes online (but not if user is already on chat page)
        if (isOnline && location.pathname !== '/chat') {
          setHighlightedNavLink('/chat');
        } else if (!isOnline && highlightedNavLink === '/chat') {
          // Remove highlight when server goes offline
          setHighlightedNavLink(null);
        }
      } catch (error) {
        setServerOnline(false);
        // Remove highlight when server is unreachable
        if (highlightedNavLink === '/chat') {
          setHighlightedNavLink(null);
        }
      }
    };

    // Initial check
    checkServerStatus();
    
    // Poll every 5 seconds
    const interval = setInterval(checkServerStatus, 5000);
    
    return () => clearInterval(interval);
  }, [location.pathname, highlightedNavLink, setHighlightedNavLink]);

  // Check for datagen inference server on port 8082
  useEffect(() => {
    const checkDatagenServerStatus = async () => {
      try {
        // Try OpenAI-compatible endpoint
        const response = await fetch('http://127.0.0.1:8082/v1/models', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        });
        
        setDatagenServerOnline(response.ok);
      } catch (error) {
        setDatagenServerOnline(false);
      }
    };

    // Initial check
    checkDatagenServerStatus();
    
    // Poll every 5 seconds
    const interval = setInterval(checkDatagenServerStatus, 5000);
    
    return () => clearInterval(interval);
  }, []);

  const navLinkClass = (path) => ({ isActive }) => {
    const base = 'px-4 py-2 rounded hover:bg-gray-700 transition-colors';
    const active = isActive ? 'bg-blue-600 text-white font-bold' : 'text-gray-300';
    const pulse = highlightedNavLink === path && !isActive ? 'animate-pulse-medium-white' : '';
    return `${base} ${active} ${pulse}`;
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      <nav className="bg-gray-800 shadow-md p-4 flex items-center gap-4">
        <NavLink to="/" className="mr-4 text-xl font-bold text-white hover:text-gray-300 flex items-center">
          <img src={atkLogo} alt="ATK Logo" className="h-10 mr-2 bg-gray-100 p-1 rounded-md" />
        </NavLink>
        <NavLink to="/configs" className={navLinkClass('/configs')}>
          Configs
        </NavLink>
        <NavLink to="/inputs" className={navLinkClass('/inputs')}>
          Inputs
        </NavLink>
        <NavLink to="/outputs" className={navLinkClass('/outputs')}>
          Outputs
        </NavLink>
        <NavLink to="/logs" className={navLinkClass('/logs')}>
          Logs
        </NavLink>
        <NavLink to="/monitor" className={navLinkClass('/monitor')}>
          Monitor
        </NavLink>
        <NavLink to="/chat" className={navLinkClass('/chat')}>
          Chat
        </NavLink>
        
        {/* Datagen Inference Server Status */}
        <div className="ml-auto flex flex-col items-end gap-1">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              datagenServerOnline ? 'bg-green-500' : 'bg-red-500'
            }`} />
            <span className="text-xs text-gray-400">
              Datagen Server
            </span>
          </div>
          <span className="text-xs text-gray-500 text-right max-w-48">
            {datagenServerOnline 
              ? 'Online (Port 8082)'
              : 'Offline - Start with local_* scripts (takes 1-2 minutes after project startup)'
            }
          </span>
        </div>
        
        {/* TODO: Add Settings/Boring Mode Toggle */}
      </nav>
      <main className="flex-grow overflow-auto">
        <Outlet /> {/* Child routes will render here */}
      </main>
      <style>{animationStyles}</style>
    </div>
  );
}

export default Layout; 