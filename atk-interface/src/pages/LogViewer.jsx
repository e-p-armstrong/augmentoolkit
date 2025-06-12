import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import Editor from '@monaco-editor/react';
import { FiChevronLeft, FiLoader, FiAlertTriangle, FiRefreshCw } from 'react-icons/fi';
import { fetchTaskLogs } from '../api'; // Import the specific API function

function LogViewer() {
  const { taskId } = useParams(); // Task ID from URL parameter
  const navigate = useNavigate();

  // State for log content, loading, and errors
  const [logContent, setLogContent] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const editorRef = useRef(null);

  // Fetch log content function
  const fetchLogs = useCallback(async (showLoadingSpinner = true) => {
    if (!taskId) {
        setError('Task ID is missing from URL.');
        setIsLoading(false);
        return;
    }
    if (showLoadingSpinner) setIsLoading(true);
    else setIsRefreshing(true); // Indicate refresh instead of initial load
    setError(null);

    try {
        // Fetch all logs initially (tailing can be added later if needed)
      const result = await fetchTaskLogs(taskId /*, optional tail parameter */);
      setLogContent(result?.logs || ''); // Use logs property from API response
    } catch (e) {
      console.error("Failed to fetch logs:", e);
      setError(e.message || `Failed to load logs for task ${taskId}.`);
      setLogContent(''); // Clear content on error
    } finally {
      if (showLoadingSpinner) setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [taskId]);

  // Fetch logs on initial mount or when taskId changes
  useEffect(() => {
    fetchLogs(true);
  }, [fetchLogs]); // fetchLogs depends on taskId

  // Handle refresh button click
  const handleRefresh = () => {
      fetchLogs(false); // Fetch logs without the main loading spinner
  };

  // Optional: Function to scroll to bottom when editor mounts or content updates
  function handleEditorDidMount(editor, monaco) {
    editorRef.current = editor;
    // Scroll to bottom after a short delay to allow rendering
    setTimeout(() => {
       editor.revealLine(editor.getModel().getLineCount());
    }, 100);
  }

  // --- Styling ---
  const buttonBaseClass = "px-4 py-2 rounded font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm h-9 flex items-center justify-center";
  const backButtonClass = "text-gray-300 hover:text-white hover:bg-gray-700";
  const refreshButtonClass = `${buttonBaseClass} bg-gray-700 hover:bg-gray-600 text-gray-200`;

  return (
    <div className="flex flex-col h-full p-4">
      {/* Header Row */}
      <div className="flex justify-between items-center mb-4 gap-4">
        {/* Left Side: Title & Back Button */}
        <div className="flex items-center gap-3 min-w-0">
          <Link
            to="/logs" // Link back to Log Manager
            className={`${buttonBaseClass} ${backButtonClass} pl-2 pr-3`}
            aria-label="Back to log manager"
          >
            <FiChevronLeft className="w-5 h-5 mr-1" /> Back
          </Link>
          <div className="min-w-0">
            <h1 className="text-xl lg:text-2xl font-bold text-gray-100 truncate">Log Viewer</h1>
            <p className="text-gray-400 text-xs lg:text-sm mt-1 truncate" title={taskId}>Viewing logs for Task ID: <code className="bg-gray-700 px-1 rounded">{taskId}</code></p>
          </div>
        </div>

        {/* Right Side: Action Buttons (e.g., Refresh) */}
         <div className="flex items-center gap-2 flex-shrink-0">
            <button
                 onClick={handleRefresh}
                 disabled={isLoading || isRefreshing}
                 className={refreshButtonClass}
                 title="Refresh log content"
            >
                 <FiRefreshCw className={`mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
                 {isRefreshing ? 'Refreshing...' : 'Refresh'}
            </button>
             {/* Add Download Button later if needed */}
         </div>
      </div>

      {/* Error Display Area */}
      {error && (
         <div className="text-red-400 bg-red-900 bg-opacity-50 p-3 rounded border border-red-700 mb-4 text-sm flex items-center gap-2">
            <FiAlertTriangle className="flex-shrink-0"/>
            <span>{error}</span>
         </div>
      )}

      {/* Editor (Viewer) Area */}
      <div className="flex-grow relative border border-gray-700 rounded overflow-hidden min-h-[200px]">
       {isLoading ? (
         <div className="absolute inset-0 flex items-center justify-center bg-gray-850">
           <FiLoader className="animate-spin text-gray-500 text-4xl" />
           <p className="text-gray-400 ml-3">Loading logs...</p>
         </div>
       ) : error && !logContent ? (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-850 text-gray-500 px-4 text-center">
             Could not load log content. Check if the log file exists and the API server is running.
          </div>
       ) : (
          <Editor
            height="100%"
            // Use 'log' language if available/registered, otherwise 'plaintext'
            language="plaintext" // Or try 'log' if you configure Monaco for it
            theme="vs-dark" // Or your preferred theme
            value={logContent}
            onMount={handleEditorDidMount} // Optional: Scroll to bottom
            options={{
              readOnly: true, // Make it read-only
              minimap: { enabled: true },
              scrollBeyondLastLine: false,
              fontSize: 13, // Slightly smaller for logs potentially
              wordWrap: 'on', // Wrap long lines
              automaticLayout: true, // Ensure editor resizes correctly
              renderWhitespace: "boundary", // Show spaces/tabs subtly if needed
              // Consider line numbers
              lineNumbers: "on",
            }}
          />
       )}
      </div>
    </div>
  );
}

export default LogViewer;
