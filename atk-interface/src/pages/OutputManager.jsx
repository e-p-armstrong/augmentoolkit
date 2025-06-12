import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Tree } from 'react-arborist';
import { FiFile, FiFolder, FiDownloadCloud, FiTrash2, FiLoader, FiAlertTriangle, FiCheckCircle, FiChevronDown, FiChevronRight } from 'react-icons/fi';
import { downloadOutputItem, deleteOutputItem, moveOutputItem } from '../api'; // Removed fetchOutputStructure
import { useFileManager } from '../context/FileManagerContext'; // Import the context hook
import Modal from '../components/Modal';
import FireworksBackground from '../components/FireworksBackground'; // Import the new background
import EnhancedButton from '../components/EnhancedButton';

/**
 * @typedef {object} OutputNodeData // Renamed type
 * @property {string} id - Unique ID (can be the relative path)
 * @property {string} name - Display name (basename of the path)
 * @property {boolean} [isInternal] - Flag for placeholder nodes
 * @property {string} relativePath - Store the full relative path
 * @property {boolean} is_dir - Keep the directory flag
 * @property {OutputNodeData[] | null} [children] - Children array or null if leaf
 */

/** @param {import("react-arborist").NodeRendererProps<OutputNodeData> & { onNodeClick: Function, lastClick: { id: string | null, time: number } }} props */
function Node({ node, style, dragHandle, onNodeClick, lastClick }) { // Added click tracking props
  const Icon = node.isLeaf ? FiFile : FiFolder;
  const iconColor = node.isLeaf ? 'text-blue-400' : 'text-yellow-500';

  const handleRowClick = (e) => {
    const now = Date.now();
    const DOUBLE_CLICK_THRESHOLD = 300; // ms

    if (lastClick.id === node.id && (now - lastClick.time) < DOUBLE_CLICK_THRESHOLD) {
      // Double click detected
      if (node.isInternal) {
        node.toggle(); // Toggle folder open/closed state
      }
      // Reset last click to prevent triple click issues
      onNodeClick(null, 0);
    } else {
      // Single click
      node.handleClick(e); // Let react-arborist handle selection
      onNodeClick(node.id, now); // Update last click state
    }
  };

  const handleCaretClick = (e) => {
    e.stopPropagation();
    node.toggle();
  };

  return (
    <div
      ref={dragHandle}
      style={style}
      className={`flex items-center gap-2 px-2 py-1 rounded cursor-pointer
                  ${node.isSelected ? 'bg-blue-700' : 'hover:bg-gray-700'}
                  ${node.isEditing ? 'bg-gray-600' : ''}`}
      onClick={handleRowClick}
    >
      {/* caret */}
      <span
        className="w-6 text-center flex items-center justify-center"
        onClick={handleCaretClick}
      >
        {node.isInternal && (node.isOpen ? <FiChevronDown /> : <FiChevronRight />)}
      </span>

      {/* icon + name */}
      <Icon className={`${iconColor} flex-shrink-0 mr-1`} />
      <span className="truncate flex-grow text-gray-200">
        {node.data.name}
      </span>
    </div>
  );
}

const GREEN_BURST = '#22FF22'; // Vivid Green
const RED_BURST = '#FF2222';   // Vivid Red

function OutputManager() {
    // Replace local state with context state
    const { outputTreeData: treeData, isOutputLoading: isLoading, outputError: error, fetchOutputData, refreshOutputData } = useFileManager();
    /** @type {[OutputNodeData[], Function]} */
    const [selectedNodes, setSelectedNodes] = useState([]);
    const treeRef = useRef(null);

    // State for manual double-click tracking
    const [lastClick, setLastClick] = useState({ id: null, time: 0 });

    // State for shake animation trigger
    const [isShaking, setIsShaking] = useState(false);

    // State for fireworks burst trigger
    const [triggerFireworksBurst, setTriggerFireworksBurst] = useState(false);

    // State for burst color
    const [burstColor, setBurstColor] = useState(null);

    const handleNodeClick = (id, time) => {
      setLastClick({ id, time });
    };

    // Modals State
    const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
    // Removed Create Folder Modal state
    const [modalError, setModalError] = useState(null);
    const [isSubmittingModal, setIsSubmittingModal] = useState(false);

    // Download state
    const [isDownloading, setIsDownloading] = useState(false);
    const [downloadStatus, setDownloadStatus] = useState({ message: '', error: false });

    // Move/Drag State
    const [isMoving, setIsMoving] = useState(false);
    const [moveError, setMoveError] = useState(null);

    // Use context fetch function
    useEffect(() => {
        fetchOutputData(); // Fetch data on mount (uses cache if available)
    }, [fetchOutputData]); // Dependency ensures it runs once on mount

    // Trigger initial firework burst on mount
    useEffect(() => {
        setBurstColor('#FFFFFF'); // Set to white
        setTriggerFireworksBurst(true);
        // Reset after a short delay
        const timer = setTimeout(() => {
            setTriggerFireworksBurst(false);
            setBurstColor(null); // Reset color if needed
        }, 500); // Adjust delay as needed

        return () => clearTimeout(timer); // Cleanup timer
    }, []); // Empty dependency array ensures this runs only once on mount

    // Effect to trigger shake animation on selection change
    useEffect(() => {
      if (selectedNodes.length > 0) {
        setIsShaking(true);
        const timer = setTimeout(() => {
          setIsShaking(false);
        }, 400); // Duration of the shake animation
        return () => clearTimeout(timer); // Cleanup timer on unmount or if selection changes again quickly
      } else {
        setIsShaking(false); // Ensure shaking stops if selection is cleared
      }
    }, [selectedNodes.length]); // Depend only on the length changing

    const handleSelect = (nodes) => {
        setSelectedNodes(nodes.map(n => n.data));
    };

    const handleToggle = (id) => {
        console.log('Toggled:', id);
    };

    // --- Move Handler ---
    /** @param {import("react-arborist").MoveHandler<OutputNodeData>} args */
    const handleMove = async ({ dragIds, parentId, index }) => {
        setIsMoving(true);
        setMoveError(null);
        console.log("Move detected:", { dragIds, parentId, index });

        const parentNode = parentId ? treeRef.current?.get(parentId) : null;
        const parentPath = parentNode ? parentNode.data.relativePath : '.';
        console.log("Parent Node:", parentNode);
        console.log("Parent Path:", parentPath);

        if (parentNode && !parentNode.data.is_dir) {
             setMoveError("Cannot move items into a file.");
             setIsMoving(false);
             return;
        }

        let successCount = 0;
        let firstError = null;

        const results = await Promise.allSettled(
            dragIds.map(id => {
                const node = treeRef.current?.get(id);
                if (!node) return Promise.reject(new Error(`Node with ID ${id} not found.`));

                const sourcePath = node.data.relativePath;
                const itemName = node.data.name;
                const cleanParentPath = parentPath === '.' ? '' : parentPath.replace(/^\/+|\/+$/g, '');
                const cleanItemName = itemName.replace(/^\/+/,'');
                const destinationPath = cleanParentPath ? `${cleanParentPath}/${cleanItemName}` : cleanItemName;

                console.log(`Attempting move: ${sourcePath} -> ${destinationPath}`);
                return moveOutputItem(sourcePath, destinationPath); // Use moveOutputItem
            })
        );

         results.forEach((result, i) => {
             const nodeId = dragIds[i];
             const node = treeRef.current?.get(nodeId);
             const itemName = node?.data?.name || `Item ${i+1}`;
            if (result.status === 'fulfilled') {
                console.log(`Successfully moved ${itemName}:`, result.value);
                successCount++;
            } else {
                console.error(`Failed to move ${itemName}:`, result.reason);
                 if (!firstError) {
                     firstError = result.reason?.message || `Failed to move ${itemName}.`;
                 }
            }
        });

        setIsMoving(false);

        if (firstError) {
            setMoveError(`Move Error: ${firstError}${successCount > 0 ? ` (${successCount} items moved successfully)` : ''}`);
        } else {
            console.log(`Successfully moved ${successCount} item(s).`);
        }

        if (successCount > 0 || firstError) {
             refreshOutputData(); // Refresh data after move
             setSelectedNodes([]);
             treeRef.current?.deselectAll();
        }
    };
    // --- End Move Handler ---

    // Removed handleCreateFolder function

    const handleDownload = async () => {
        if (selectedNodes.length === 0 || isDownloading) return;

        setIsDownloading(true);
        setDownloadStatus({ message: `Starting download for ${selectedNodes.length} item(s)...`, error: false });
        setBurstColor(GREEN_BURST); // Set green color for burst
        setTriggerFireworksBurst(true); // Trigger burst effect

        let successCount = 0;
        let firstError = null;

        for (const node of selectedNodes) {
            try {
                setDownloadStatus({ message: `Downloading ${node.name}...`, error: false });
                const response = await downloadOutputItem(node.relativePath); // Use downloadOutputItem
                const blob = await response.blob();

                let filename = node.name;
                const disposition = response.headers.get('content-disposition');
                if (disposition && disposition.indexOf('attachment') !== -1) {
                    const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
                    const matches = filenameRegex.exec(disposition);
                    if (matches != null && matches[1]) {
                        filename = matches[1].replace(/['"]/g, '');
                    }
                }
                if (node.is_dir && !filename.toLowerCase().endsWith('.zip')) {
                    filename += '.zip';
                }

                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                a.remove();
                successCount++;
                 await new Promise(resolve => setTimeout(resolve, 200));
            } catch (error) {
                console.error(`Failed to download ${node.relativePath}:`, error);
                if (!firstError) {
                    firstError = error.message || `Failed to download ${node.name}.`;
                }
            }
        }

        setIsDownloading(false);
        if (firstError) {
            setDownloadStatus({ message: `Download failed: ${firstError}`, error: true });
        } else {
            setDownloadStatus({ message: `Successfully downloaded ${successCount} item(s).`, error: false });
        }
        // Reset burst trigger and color shortly after download attempt finishes
        setTimeout(() => {
            setTriggerFireworksBurst(false);
            setBurstColor(null);
        }, 500);
        setTimeout(() => setDownloadStatus({ message: '', error: false }), 5000);
    };

    const handleDelete = async () => {
        if (selectedNodes.length === 0 || isSubmittingModal) return;

        const itemsToDelete = [...selectedNodes];
        // Trigger red burst *before* starting async operation
        setBurstColor(RED_BURST);
        setTriggerFireworksBurst(true);

        setIsSubmittingModal(true);
        setModalError(null);
        let successCount = 0;
        let firstError = null;

        const results = await Promise.allSettled(
            itemsToDelete.map(node => deleteOutputItem(node.relativePath)) // Use deleteOutputItem
        );

        results.forEach((result, index) => {
            if (result.status === 'fulfilled') {
                successCount++;
            } else {
                console.error(`Failed to delete ${itemsToDelete[index].relativePath}:`, result.reason);
                if (!firstError) {
                    firstError = result.reason?.message || `Failed to delete ${itemsToDelete[index].name}.`;
                }
            }
        });

        setIsSubmittingModal(false);

        if (firstError) {
            setModalError(`Error: ${firstError}${successCount > 0 ? ` (${successCount} items deleted successfully)` : ''}`);
        } else {
            console.log(`Successfully deleted ${successCount} items.`);
            setIsDeleteModalOpen(false);
        }

        // Reset burst trigger and color after operation completes
        setTimeout(() => {
            setTriggerFireworksBurst(false);
            setBurstColor(null);
        }, 500);

        if (successCount > 0) {
            refreshOutputData(); // Refresh data after deletion
            setSelectedNodes([]);
        }
    };

    // NOTE to self -- we will have to cache the darned outputs/inputs once the interface is started, since they will often be quite big.

    // Removed openCreateFolderModal function

    const openDeleteModal = () => {
        setModalError(null);
        setIsDeleteModalOpen(true);
    }

    const actionButtonClass = "flex items-center gap-2 px-3 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm transition-colors duration-200 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed";

    // Define the background click handler similar to ConfigSelector/InputManager
    const handleBackgroundClick = useCallback((e) => {
        // Don't clear if a modal is open
        if (isDeleteModalOpen) return;

        const clickedOnRow = e.target.closest('[role="treeitem"]');
        const clickedOnScrollbar = e.target.closest('.simplebar-scrollbar'); // Adjust if needed
        const clickedOnActionButton = e.target.closest('[data-action-button="true"]');
        const clickedOnModal = e.target.closest('[role="dialog"]');

        if (!clickedOnRow && !clickedOnScrollbar && !clickedOnActionButton && !clickedOnModal) {
            if (treeRef.current?.deselectAll) {
                treeRef.current.deselectAll();
            }
            if (typeof setSelectedNodes === 'function') {
                setSelectedNodes([]);
            }
        }
    }, [isDeleteModalOpen]);

    // Determine firework launch interval
    const fireworkLaunchInterval = selectedNodes.length > 0 ? 2000 : 5000; // Faster when selected

    return (
        <div className='flex flex-col h-full relative' onMouseDown={handleBackgroundClick}>
            {/* Add Fireworks Background */}
            <FireworksBackground
                launchInterval={fireworkLaunchInterval}
                triggerBurst={triggerFireworksBurst}
                burstColorOverride={burstColor} // Pass the color override
            />

            {/* Main Content Area (Wrapped and Centered) */}
            <div className="relative z-10 flex-grow flex flex-col max-w-7xl w-full mx-auto p-6">

                {/* Header - Apply background, blur, padding, border */}
                <div className="flex justify-between items-center mb-4 bg-gray-900 bg-opacity-30 backdrop-blur-sm p-4 rounded-t-lg border-b border-gray-700">
                    <h1 className="text-2xl font-bold text-gray-100">Output Manager</h1>
                    <div className="flex items-center gap-2">
                        <EnhancedButton
                            onClick={handleDownload}
                            className={`${actionButtonClass}
                                       ${selectedNodes.length > 0 && !isDownloading ? 'animate-pulse-green-download' : 'hover:bg-green-700'}
                                       ${isShaking ? 'animate-shake' : ''}`}
                            title="Download Selected Item(s)"
                            disabled={selectedNodes.length === 0 || isDownloading}
                            data-action-button="true"
                            particleColor="#34d399"
                        >
                            {isDownloading ? <FiLoader className="animate-spin"/> : <FiDownloadCloud />}
                            {isDownloading ? 'Downloading...' : 'Download'}
                        </EnhancedButton>
                        <EnhancedButton
                          onClick={openDeleteModal}
                          className={`${actionButtonClass}
                                     ${selectedNodes.length > 0 ? 'animate-pulse-red' : 'text-red-400 hover:bg-red-900 hover:text-red-300'}
                                     ${isShaking ? 'animate-shake' : ''}`}
                          title="Delete Selected Item(s)"
                          disabled={selectedNodes.length === 0}
                          data-action-button="true"
                          particleColor="#ef4444"
                        >
                            <FiTrash2 /> Delete
                        </EnhancedButton>
                    </div>
                </div>

                 {/* Loading/Error States for Initial Load - Apply background */}
                {isLoading && !treeData && (
                    <div className="flex-grow flex items-center justify-center text-gray-400 bg-gray-900 bg-opacity-50 backdrop-blur-sm p-4 rounded-b-lg">
                        <FiLoader className="animate-spin mr-3" /> Loading output files...
                    </div>
                )}
                {error && !treeData && (
                    <div className="flex-grow flex flex-col items-center justify-center text-red-400 bg-red-900 bg-opacity-60 backdrop-blur-sm p-6 rounded-b-lg border border-red-700">
                        <FiAlertTriangle className="w-10 h-10 mb-2" />
                        <p>Error loading output structure:</p>
                        <p className="text-sm mt-1">{error}</p>
                        <button onClick={() => fetchOutputData(true)} className="mt-4 px-3 py-1.5 rounded bg-gray-600 hover:bg-gray-500 text-gray-200 text-sm">
                            Retry
                        </button>
                    </div>
                )}

                {/* Tree View Area - Apply background */}            
                {treeData && (
                     <div className={`flex-grow border-b border-r border-l border-gray-700 rounded-b-lg bg-gray-900 bg-opacity-30 backdrop-blur-sm overflow-auto relative min-h-0 ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}>                    
                         {/* Loading overlay during refresh */}                     
                         {isLoading && (
                             <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-50 z-10">
                                 <FiLoader className="animate-spin text-gray-400 text-2xl" />
                             </div>
                         )}
                         {/* Tree or Empty State */}                    
                         {(treeData && treeData.length > 0) ? (
                             <Tree
                                 ref={treeRef}
                                 data={treeData}
                                 openByDefault={false}
                                 width="100%"
                                 height={1000} // Use large height, container scrolls
                                 indent={24}
                                 rowHeight={32}
                                 paddingTop={10}
                                 paddingBottom={10}
                                 onSelect={handleSelect}
                                 onToggle={handleToggle}
                                 onMove={handleMove}
                                 disableDrop={({ parentNode, dropTarget }) => {
                                   return dropTarget && !dropTarget.data.is_dir;
                                 }}
                                 children={(props) => <Node {...props} onNodeClick={handleNodeClick} lastClick={lastClick} />}                              >
                             </Tree>
                         ) : (
                             <div className="flex flex-col items-center justify-center h-full text-gray-500">
                                <p>Output directory is empty.</p>
                            </div>
                         )}
                    </div>
                )}

                {/* Status Bar Area - Apply background */}
                <div className="h-8 mt-2 text-sm flex items-center gap-4 overflow-hidden whitespace-nowrap px-4 py-1 bg-gray-900 bg-opacity-80 backdrop-blur-sm rounded-lg border border-gray-700">

                    {/* Download Status */}                    
                    {isDownloading && (
                         <div className="flex items-center text-blue-400 flex-shrink-0">
                            <FiLoader className="animate-spin mr-2 flex-shrink-0" /> {downloadStatus.message}
                         </div>
                    )}
                    {!isDownloading && downloadStatus.message && (
                       <p className={`${downloadStatus.error ? 'text-red-400' : 'text-green-400'} truncate`} title={downloadStatus.message}>{downloadStatus.message}</p>
                    )}

                    {/* Move Status */}                    
                     {isMoving && (
                        <div className="flex items-center text-blue-400 flex-shrink-0">
                           <FiLoader className="animate-spin mr-2 flex-shrink-0" /> Moving items...
                        </div>
                    )}
                    {moveError && <p className="text-red-400 truncate" title={moveError}>Move Error: {moveError}</p>}

                </div>

            </div> { /* End Main Content Wrapper */ }

            {/* --- Modals --- */}            
            {/* Modals remain outside the centered content */}
            <Modal
                isOpen={isDeleteModalOpen}
                onClose={() => !isSubmittingModal && setIsDeleteModalOpen(false)}
                title="Confirm Deletion"
            >
                <p className="mb-4">Are you sure you want to delete the following {selectedNodes.length} item(s)? This action cannot be undone.</p>
                <ul className="list-disc list-inside mb-4 max-h-40 overflow-y-auto bg-gray-700 p-2 rounded border border-gray-600">
                    {selectedNodes.map(node => <li key={node.id} className="truncate text-sm py-0.5" title={node.relativePath}>{node.relativePath}</li>)}
                </ul>
                {modalError && (
                    <p className="text-red-400 text-sm mb-3 bg-red-900 bg-opacity-40 p-2 rounded border border-red-700">{modalError}</p>
                )}
                 <div className="flex justify-end gap-3 mt-4">
                    <button
                        onClick={() => setIsDeleteModalOpen(false)}
                        className="px-4 py-2 rounded bg-gray-600 hover:bg-gray-500 text-gray-200"
                        disabled={isSubmittingModal}
                     >
                         Cancel
                    </button>
                    <EnhancedButton
                        onClick={handleDelete}
                        className={`px-4 py-2 rounded bg-red-600 hover:bg-red-500 text-white flex items-center disabled:opacity-50
                                    ${isDeleteModalOpen && !isSubmittingModal && selectedNodes.length > 0 ? 'animate-pulse-red' : ''}`}
                        disabled={isSubmittingModal || selectedNodes.length === 0}
                        particleColor="#ef4444"
                    >
                       {isSubmittingModal ? <FiLoader className="animate-spin mr-2" /> : <FiTrash2 className="mr-1" />}
                       {isSubmittingModal ? 'Deleting...' : 'Delete'}
                    </EnhancedButton>
                 </div>
            </Modal>

            {/* Removed Create Folder Modal */}

        </div>
    );
}

export default OutputManager; 