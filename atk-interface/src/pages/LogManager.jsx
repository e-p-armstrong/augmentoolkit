import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import { Tree } from 'react-arborist'; // Import Tree
import { FiTrash2, FiRefreshCw, FiList, FiLoader, FiAlertTriangle, FiCheckCircle, FiSlash, FiFileText, FiChevronRight } from 'react-icons/fi'; // Add FiFileText, FiChevronRight
import { fetchLogFiles, deleteLogFile, clearAllLogs } from '../api';
import Modal from '../components/Modal';
import DocumentStackBackground from '../components/DocumentStackBackground'; // Import the new background
import EnhancedButton from '../components/EnhancedButton';

// --- Type Definition for react-arborist ---
/**
 * @typedef {object} LogNodeData
 * @property {string} id - Unique ID (the filename)
 * @property {string} name - Display name (the filename)
 * @property {string} taskId - Task ID extracted from filename
 */

// --- Node Renderer Component ---
/** @param {import("react-arborist").NodeRendererProps<LogNodeData> & { navigate: Function, onNodeClick: Function, lastClick: { id: string | null, time: number } }} props */
function Node({ node, style, dragHandle, navigate, onNodeClick, lastClick }) {
  const Icon = FiFileText; // Always a file for logs
  const iconColor = 'text-gray-400'; // Example color for logs

  const handleRowClick = (e) => {
    const now = Date.now();
    const DOUBLE_CLICK_THRESHOLD = 300; // ms

    if (lastClick.id === node.id && (now - lastClick.time) < DOUBLE_CLICK_THRESHOLD) {
      // Double click detected - navigate to log viewer
      if (node.data.taskId) {
        navigate(`/logs/${encodeURIComponent(node.data.taskId)}`); // Pass task ID
      } else {
        console.warn("Cannot navigate: Task ID missing for node", node.data.name);
      }
      // Reset last click
      onNodeClick(null, 0);
    } else {
      // Single click
      node.handleClick(e); // Let react-arborist handle selection
      onNodeClick(node.id, now); // Update last click state
    }
  };

  // Logs are always leaves, so no caret logic needed here

  return (
    <div
      ref={dragHandle} // Enable drag selection if needed, but disable drop
      style={style}
      className={`flex items-center gap-2 px-2 py-1 rounded cursor-pointer
                  ${node.isSelected ? 'bg-blue-700' : 'hover:bg-gray-700'}
                  ${node.isEditing ? 'bg-gray-600' : ''}`} // Should not be editable
      onClick={handleRowClick} // Use single click handler for selection/navigation
      title={`Double-click to view ${node.data.name}`} // Tooltip
    >
      {/* Placeholder for caret space, keeps alignment */}
      <span className="w-6 text-center flex items-center justify-center">
        {/* No caret for files */}
      </span>

      {/* Icon + Name */}
      <Icon className={`${iconColor} flex-shrink-0 mr-1`} />
      <span className="truncate flex-grow text-gray-200 font-mono text-sm">
        {node.data.name}
      </span>
    </div>
  );
}


function LogManager() {
    const [treeData, setTreeData] = useState([]); // Use treeData state
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    /** @type {[LogNodeData[], Function]} */
    const [selectedNodes, setSelectedNodes] = useState([]); // Store selected node data
    const [isDeleting, setIsDeleting] = useState(false);
    const [isClearing, setIsClearing] = useState(false);
    const [actionStatus, setActionStatus] = useState({ message: '', error: false, success: false });
    const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
    const [isClearAllModalOpen, setIsClearAllModalOpen] = useState(false);
    const [modalError, setModalError] = useState(null);
    const [clearAllResult, setClearAllResult] = useState(null);
    const treeRef = useRef(null); // Ref for tree instance
    const navigate = useNavigate(); // Hook for navigation
    const [lastClick, setLastClick] = useState({ id: null, time: 0 }); // For double-click tracking

    // State for background animation triggers
    const [triggerTossAnim, setTriggerTossAnim] = useState(false);
    const [triggerExplodeAnim, setTriggerExplodeAnim] = useState(false);


    // --- Helper Functions ---
    const extractTaskId = (filename) => {
        if (typeof filename === 'string' && filename.endsWith('.log')) {
            return filename.slice(0, -4);
        }
        console.warn(`Could not extract task ID from filename: ${filename}`);
        return null;
    };

    // --- Data Fetching & Adaptation ---
    const fetchLogs = useCallback(async (showLoading = true) => {
        if (showLoading) setIsLoading(true);
        setError(null);
        setActionStatus({ message: '', error: false, success: false });
        setModalError(null);
        setClearAllResult(null);
        try {
            let files = await fetchLogFiles();
            files.sort((a, b) => b.localeCompare(a)); // Sort descending

            // Adapt data for react-arborist
            const adaptedData = files.map(filename => {
                const taskId = extractTaskId(filename);
                return {
                    id: filename, // Use filename as unique ID
                    name: filename,
                    taskId: taskId,
                };
            });
            setTreeData(adaptedData);

        } catch (e) {
            console.error("Failed to fetch log files:", e);
            setError(e.message || 'Failed to load log files. Is the API server running?');
            setTreeData([]); // Set empty on error
        } finally {
            if (showLoading) setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchLogs();
    }, [fetchLogs]);

    // --- Stable callbacks for background animations ---
    const handleExplodeComplete = useCallback(() => {
        setTriggerExplodeAnim(false);
    }, []); // No dependencies needed as setTriggerExplodeAnim is stable

    const handleTossComplete = useCallback(() => {
        setTriggerTossAnim(false);
    }, []); // No dependencies needed as setTriggerTossAnim is stable

    // --- Event Handlers ---

    // Handle node selection from react-arborist
    const handleSelect = (nodes) => {
        setSelectedNodes(nodes.map(n => n.data)); // Store the data object
    };

    // Double click tracking
    const handleNodeClick = (id, time) => {
        setLastClick({ id, time });
    };

    const handleRefresh = () => {
        setSelectedNodes([]); // Clear selection on refresh
        treeRef.current?.deselectAll(); // Deselect in tree
        fetchLogs(true); // Show loading indicator during manual refresh
    };

    const handleDeleteSelected = async () => {
        if (selectedNodes.length === 0 || isDeleting || isClearing) return;

        const filesToDelete = selectedNodes.map(node => ({ filename: node.name, taskId: node.taskId })); // Get filenames and task IDs

        // Trigger background animation *before* starting async operation
        setTriggerTossAnim(true);

        setIsDeleting(true);
        setModalError(null);
        setActionStatus({ message: '', error: false, success: false });
        let successCount = 0;
        let firstError = null;

        const results = await Promise.allSettled(
            filesToDelete.map(file => {
                if (!file.taskId) {
                    return Promise.reject(new Error(`Invalid filename format: ${file.filename}`));
                }
                return deleteLogFile(file.taskId); // Use taskId from selected node data
            })
        );

        results.forEach((result, index) => {
            const file = filesToDelete[index];
            if (result.status === 'fulfilled') {
                successCount++;
                console.log(`Successfully deleted ${file.filename}:`, result.value);
            } else {
                console.error(`Failed to delete ${file.filename}:`, result.reason);
                if (!firstError) {
                    firstError = result.reason?.message || `Failed to delete ${file.filename}.`;
                }
            }
        });

        setIsDeleting(false);

        if (firstError) {
            setModalError(`Error: ${firstError}${successCount > 0 ? ` (${successCount} log(s) deleted successfully)` : ''}`);
        } else {
            setActionStatus({ message: `Successfully deleted ${successCount} log file(s).`, error: false, success: true });
            setIsDeleteModalOpen(false); // Close modal on full success
            setTimeout(() => setActionStatus({ message: '', error: false, success: false }), 5000);
        }

        // Refresh log list if any deletions succeeded
        if (successCount > 0) {
            setSelectedNodes([]); // Clear selection state
            treeRef.current?.deselectAll(); // Deselect in tree instance
            await fetchLogs(false); // Refresh list without full loading indicator
        }
        // Reset trigger after a short delay (allows animation to start)
        // setTimeout(() => setTriggerTossAnim(false), 100); // Resetting is handled by the callback now
    };

    const handleClearAll = async () => {
        if (isClearing || isDeleting) return;

        // Trigger background animation *before* starting async operation
        setTriggerExplodeAnim(true);

        setIsClearing(true);
        setModalError(null);
        setClearAllResult(null);
        setActionStatus({ message: '', error: false, success: false });

        try {
            const result = await clearAllLogs();
            setClearAllResult(result); // Store the result { message, errors }
             // Display the primary message from the API response
            setActionStatus({ message: result.message || 'Cleared logs.', error: result.errors && result.errors.length > 0, success: !result.errors || result.errors.length === 0 });
            if (!result.errors || result.errors.length === 0) {
                 setIsClearAllModalOpen(false); // Close modal only if no errors reported by API
                 setTimeout(() => setActionStatus({ message: '', error: false, success: false }), 5000);
            } else {
                // If there were errors, keep the modal open and display them via clearAllResult
                setModalError(`Clear operation reported ${result.errors.length} error(s). See details below.`);
            }
            setSelectedNodes([]); // Clear selection
            treeRef.current?.deselectAll();
            await fetchLogs(false); // Refresh list
        } catch (e) {
            console.error("Failed to clear all logs:", e);
            // Display error within the modal
            setModalError(e.message || 'Could not clear all log files.');
             // Also update general status bar temporarily
            setActionStatus({ message: `Error clearing logs: ${e.message}`, error: true, success: false });
            setTimeout(() => setActionStatus({ message: '', error: false, success: false }), 8000);
        } finally {
            setIsClearing(false);
            // Reset trigger after a short delay (allows animation to start)
            // setTimeout(() => setTriggerExplodeAnim(false), 100); // Resetting is handled by the callback now
        }
    };

    // --- Modal Controls ---
    const openDeleteModal = () => {
        setModalError(null);
        setIsDeleteModalOpen(true);
    }

    const openClearAllModal = () => {
        setModalError(null);
        setClearAllResult(null);
        setIsClearAllModalOpen(true);
    }

     // Function to clear selection when clicking outside the tree or action buttons
     const handleBackgroundClick = useCallback((e) => {
         // Don't clear if a modal is open
         if (isDeleteModalOpen || isClearAllModalOpen) return;

         // Don't clear if the click originated from the document stack background
         const clickedOnBackground = e.target.closest('[data-id="document-stack-background"]');

         const clickedOnRow = e.target.closest('[role="treeitem"]');
         const clickedOnScrollbar = e.target.closest('.simplebar-scrollbar'); // Adjust if using a specific scrollbar library
         const clickedOnActionButton = e.target.closest('[data-action-button="true"]');
         const clickedOnModal = e.target.closest('[role="dialog"]'); // Check if click is inside any modal

         if (!clickedOnRow && !clickedOnScrollbar && !clickedOnActionButton && !clickedOnModal && !clickedOnBackground) {
             if (treeRef.current?.deselectAll) {
                 treeRef.current.deselectAll();
             }
             if (typeof setSelectedNodes === 'function') {
                 setSelectedNodes([]);
             }
         }
     }, [isDeleteModalOpen, isClearAllModalOpen]);


    // --- Render ---
    const actionButtonClass = "flex items-center gap-2 px-3 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed";

    return (
        <div className="flex flex-col h-full p-4 relative" onMouseDown={handleBackgroundClick}> {/* Added relative positioning */}
            <DocumentStackBackground
                triggerExplode={triggerExplodeAnim}
                triggerTossTop={triggerTossAnim}
                onExplodeComplete={handleExplodeComplete} // Use stable callback
                onTossComplete={handleTossComplete}      // Use stable callback
            />
            {/* Background Animation */}

            {/* Content Wrapper (needs z-index) */}
            <div className="relative z-10 flex flex-col flex-grow">

            {/* Header */}
                <div className="flex justify-between items-center mb-4 bg-gray-900 bg-opacity-50 backdrop-blur-sm p-4 rounded-t-lg border-b border-gray-700">
                <h1 className="text-2xl font-bold text-gray-100 flex items-center gap-2"><FiList /> Log Manager</h1>
                <div className="flex items-center gap-2">
                    <button
                        onClick={handleRefresh}
                        className={actionButtonClass}
                        title="Refresh log list"
                        disabled={isLoading || isDeleting || isClearing}
                        data-action-button="true"
                    >
                        <FiRefreshCw className={isLoading ? 'animate-spin' : ''} /> Refresh
                    </button>
                    <EnhancedButton
                        onClick={openDeleteModal}
                        className={`${actionButtonClass} text-orange-400 hover:bg-orange-900 hover:text-orange-300
                                        ${selectedNodes.length > 0 && !isDeleting && !isClearing ? 'animate-pulse-orange' : ''} `} // Added pulse animation
                        title="Delete Selected Log Files"
                        disabled={selectedNodes.length === 0 || isLoading || isDeleting || isClearing}
                        data-action-button="true"
                        particleColor="#fb923c"
                    >
                        <FiTrash2 /> Delete Selected ({selectedNodes.length})
                    </EnhancedButton>
                    <EnhancedButton
                        onClick={openClearAllModal}
                        className={`${actionButtonClass} text-red-400 hover:bg-red-900 hover:text-red-300
                                        ${treeData.length > 0 && selectedNodes.length === 0 && !isDeleting && !isClearing ? 'animate-pulse-red-intense' : ''} `} // MODIFIED: Added selectedNodes.length === 0 condition for pulse
                        title="Delete ALL Log Files"
                        disabled={treeData.length === 0 || isLoading || isDeleting || isClearing}
                        data-action-button="true"
                        particleColor="#ef4444"
                    >
                        <FiSlash /> Clear All Logs
                    </EnhancedButton>i
                </div>
            </div>

            {/* Loading/Error States for Initial Load */}
            {isLoading && (
                    <div className="flex-grow flex items-center justify-center text-gray-400 bg-gray-900 bg-opacity-70 backdrop-blur-sm p-4 rounded-b-lg">
                    <FiLoader className="animate-spin mr-3" /> Loading log files...
                </div>
            )}
            {!isLoading && error && (
                    <div className="flex-grow flex flex-col items-center justify-center text-red-400 bg-red-900 bg-opacity-60 backdrop-blur-sm p-6 rounded-b-lg border border-red-700">
                    <FiAlertTriangle className="w-10 h-10 mb-2" />
                    <p>Error loading log files:</p>
                    <p className="text-sm mt-1">{error}</p>
                    <button onClick={handleRefresh} className="mt-4 px-3 py-1.5 rounded bg-gray-600 hover:bg-gray-500 text-gray-200 text-sm">
                        Retry
                    </button>
                </div>
            )}
            {/* Tree View Area */}
            {!isLoading && !error && (
                    <div className="flex-grow border border-gray-700 rounded-b-lg bg-gray-900 bg-opacity-50 backdrop-blur-sm overflow-auto relative min-h-[200px]">
                     {treeData.length > 0 ? (
                        <Tree
                            ref={treeRef}
                            data={treeData}
                            openByDefault={false} // Not applicable for flat list
                            width="100%"
                            height={600} // Use a fixed numeric height
                            // Make height dynamic based on container? Or set fixed like other pages?
                            // height={600} // Fixed height example
                            indent={24}
                            rowHeight={32}
                            paddingTop={10}
                            paddingBottom={10}
                            onSelect={handleSelect}
                            disableDrop={true} // Cannot drop onto logs
                            disableDrag={true} // Cannot drag logs (no move endpoint)
                            // Allow multiple selection (default behavior often includes shift/ctrl/meta keys)
                            // selectionFollowsFocus={true} // Consider if needed for keyboard nav + selection
                        >
                            {/* Pass props needed by Node, including click tracking and navigate */}
                            {(props) => <Node {...props} navigate={navigate} onNodeClick={handleNodeClick} lastClick={lastClick} />}
                        </Tree>
                     ) : (
                         <div className="flex flex-col items-center justify-center h-full text-gray-500">
                            <FiList className="w-12 h-12 mb-4 text-gray-600" />
                            <p>No log files found in <code className="text-xs bg-gray-600 px-1 rounded">./logs/</code> directory.</p>
                        </div>
                     )}
                </div>
            )}

            {/* Status Bar Area */}
                <div className="h-8 mt-2 text-sm flex items-center gap-4 overflow-hidden whitespace-nowrap px-4 py-1 bg-gray-900 bg-opacity-80 backdrop-blur-sm rounded-lg border border-gray-700">
                {/* Deleting Status */}
                {isDeleting && (
                    <div className="flex items-center text-orange-400">
                        <FiLoader className="animate-spin mr-2 flex-shrink-0" /> Deleting selected logs...
                    </div>
                )}
                {/* Clearing Status */}
                {isClearing && (
                    <div className="flex items-center text-red-400">
                        <FiLoader className="animate-spin mr-2 flex-shrink-0" /> Clearing all logs...
                    </div>
                )}
                 {/* General Action Status */}
                 {actionStatus.message && !isDeleting && !isClearing && (
                     <p className={`${actionStatus.error ? 'text-red-400' : actionStatus.success ? 'text-green-400' : 'text-gray-400'} truncate flex items-center`} title={actionStatus.message}>
                         {actionStatus.success && <FiCheckCircle className="mr-1 flex-shrink-0"/>}
                         {actionStatus.error && <FiAlertTriangle className="mr-1 flex-shrink-0"/>}
                         {actionStatus.message}
                     </p>
                 )}
                </div>
            </div>

            {/* --- Modals --- */}

            {/* Delete Confirmation Modal */}
            <Modal
                isOpen={isDeleteModalOpen}
                onClose={() => !(isDeleting || isClearing) && setIsDeleteModalOpen(false)}
                title={`Confirm Delete ${selectedNodes.length} Log File(s)`} // Use selectedNodes.length
            >
                <p className="mb-4">Are you sure you want to permanently delete the selected {selectedNodes.length} log file(s)?</p>
                <ul className="list-disc list-inside mb-4 max-h-40 overflow-y-auto bg-gray-700 p-2 rounded border border-gray-600 font-mono text-xs">
                    {/* Use selectedNodes for list */}
                    {selectedNodes.sort((a, b) => a.name.localeCompare(b.name)).map(node => <li key={node.id} className="truncate py-0.5">{node.name}</li>)}
                </ul>
                {modalError && (
                    <p className="text-red-400 text-sm mb-3 bg-red-900 bg-opacity-40 p-2 rounded border border-red-700">{modalError}</p>
                )}
                <div className="flex justify-end gap-3 mt-4">
                    <button
                        onClick={() => setIsDeleteModalOpen(false)}
                        className="px-4 py-2 rounded bg-gray-600 hover:bg-gray-500 text-gray-200"
                        disabled={isDeleting || isClearing}
                    >
                        Cancel
                    </button>
                    <EnhancedButton
                        onClick={handleDeleteSelected}
                        className={`px-4 py-2 rounded bg-orange-600 hover:bg-orange-500 text-white flex items-center disabled:opacity-50
                                    ${isDeleteModalOpen && !isDeleting && !isClearing && selectedNodes.length > 0 ? 'animate-pulse-orange' : ''}`}
                        disabled={isDeleting || isClearing || selectedNodes.length === 0} // Use selectedNodes.length
                        particleColor="#fb923c"
                    >
                        {isDeleting ? <FiLoader className="animate-spin mr-2" /> : <FiTrash2 className="mr-1" />}
                        {isDeleting ? 'Deleting...' : `Delete Selected`}
                    </EnhancedButton>
                </div>
            </Modal>

            {/* Clear All Confirmation Modal */}
            <Modal
                isOpen={isClearAllModalOpen}
                onClose={() => !(isDeleting || isClearing) && setIsClearAllModalOpen(false)}
                title="Confirm Clear All Logs"
            >
                 {/* Use treeData.length for total count */}
                <p className="mb-4">Are you sure you want to permanently delete <strong>ALL</strong> ({treeData.length}) log files in the <code className="text-xs bg-gray-600 px-1 rounded">./logs/</code> directory? This action cannot be undone.</p>
                 {/* Display API-reported errors if any occurred during the clear operation */}
                 {clearAllResult?.errors && clearAllResult.errors.length > 0 && (
                     <div className="mb-4">
                         <p className="text-yellow-400 text-sm mb-2">The following errors occurred during the clear operation:</p>
                         <ul className="list-disc list-inside max-h-32 overflow-y-auto bg-gray-700 p-2 rounded border border-gray-600 text-xs">
                             {clearAllResult.errors.map((err, index) => (
                                 <li key={index} className="truncate py-0.5">{typeof err === 'string' ? err : JSON.stringify(err)}</li>
                             ))}
                         </ul>
                     </div>
                 )}
                 {/* General modal error display */}
                 {modalError && (
                    <p className={`text-sm mb-3 bg-opacity-40 p-2 rounded border ${clearAllResult?.errors?.length > 0 ? 'bg-yellow-900 border-yellow-700 text-yellow-300' : 'bg-red-900 border-red-700 text-red-300' }`}>{modalError}</p>
                 )}
                 <div className="flex justify-end gap-3 mt-4">
                    <button
                        onClick={() => setIsClearAllModalOpen(false)}
                        className="px-4 py-2 rounded bg-gray-600 hover:bg-gray-500 text-gray-200"
                        disabled={isDeleting || isClearing}
                    >
                        {clearAllResult?.errors?.length > 0 ? 'Close' : 'Cancel'} {/* Change label if errors occurred */}
                    </button>
                    {/* Only show the 'Clear All' button if no errors occurred or before the attempt */}
                    {(!clearAllResult || !clearAllResult.errors || clearAllResult.errors.length === 0) && (
                        <EnhancedButton
                            onClick={handleClearAll}
                            className={`px-4 py-2 rounded bg-red-600 hover:bg-red-500 text-white flex items-center disabled:opacity-50
                                        ${isClearAllModalOpen && !isDeleting && !isClearing && treeData.length > 0 ? 'animate-pulse-red-intense' : ''}`}
                             // Use treeData.length for disable check
                            disabled={isDeleting || isClearing || treeData.length === 0}
                            particleColor="#ef4444"
                        >
                            {isClearing ? <FiLoader className="animate-spin mr-2" /> : <FiSlash className="mr-1" />}
                            {isClearing ? 'Clearing...' : `Clear All Logs`}
                        </EnhancedButton>
                     )}
                 </div>
            </Modal>

        </div>
    );
}

export default LogManager;