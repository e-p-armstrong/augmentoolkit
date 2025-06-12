import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Tree } from 'react-arborist';
import { FiFile, FiFolder, FiFolderPlus, FiCopy, FiTrash2, FiLoader, FiAlertTriangle, FiCheckCircle, FiChevronDown, FiChevronRight } from 'react-icons/fi';
import { fetchConfigStructure, deleteConfigItem, moveConfigItem, createConfigDirectory, duplicateConfig, fetchAvailableConfigAliases } from '../api'; // Added new API functions
import Modal from '../components/Modal';
import ScrollingTextBackground from '../components/ScrollingTextBackground'; // Import the background component
import EnhancedButton from '../components/EnhancedButton'; // Import the new button component

// --- CSS Keyframes for Animations ---
// NOTE: Ideally, move these to a global CSS file or tailwind.config.js
// const animationStyles = `
//   @keyframes shake {
//     0%, 100% { transform: translateX(0); }
//     10%, 30%, 50%, 70%, 90% { transform: translateX(-2px); }
//     20%, 40%, 60%, 80% { transform: translateX(2px); }
//   }
//   .animate-shake {
//     animation: shake 0.4s ease-in-out; /* Shorter duration, run once per selection change */
//   }
//
//   @keyframes pulse-red {
//     0%, 100% { background-color: #dc2626; color: white; } /* bg-red-600 text-white */
//     50% { background-color: #374151; color: #f87171; } /* bg-gray-700 text-red-400 */
//   }
//   .animate-pulse-red {
//     animation: pulse-red 1.2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
//   }
//
//   @keyframes pulse-white {
//     0%, 100% { background-color: rgba(255, 255, 255, 0.1); } /* Slightly white background */
//     50% { background-color: rgba(255, 255, 255, 0.3); } /* More prominent white */
//   }
//   .animate-pulse-white {
//       /* Apply this animation on top of other background styles */
//       animation: pulse-white 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
//   }
//
//   /* Additional subtle pulse animations for guidance */
//   @keyframes pulse-bg-subtle {
//     50% { background-color: rgba(75, 85, 99, 0.8); } /* bg-gray-600 slightly lighter */
//   }
//   .animate-pulse-bg-subtle {
//       animation: pulse-bg-subtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
//       background-color: rgba(255, 255, 255, 0.5); /* More intense white color */
//   }
//
//   @keyframes pulse-border-subtle {
//     50% { border-color: rgba(107, 114, 128, 0.8); } /* border-gray-500 slightly lighter */
//   }
//   .animate-pulse-border-subtle {
//       animation: pulse-border-subtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
//   }
//
//   @keyframes pulse-blue-gold {
//       0% { background-color: #2563eb; box-shadow: 0 0 0 0 rgba(251, 191, 36, 0.7); } /* bg-blue-600, shadow-amber-400 */
//       70% { background-color: #3b82f6; box-shadow: 0 0 10px 10px rgba(251, 191, 36, 0); } /* bg-blue-500 */
//       100% { background-color: #2563eb; box-shadow: 0 0 0 0 rgba(251, 191, 36, 0); }
//   }
//   .animate-pulse-blue-gold {
//       animation: pulse-blue-gold 2s infinite;
//   }
// `;

// --- Type Definitions ---

/**
 * @typedef {object} ConfigNodeData
 * @property {string} id - Unique ID (relative path)
 * @property {string} name - Display name (basename)
 * @property {boolean} [isInternal] - react-arborist internal node flag
 * @property {string} relativePath - Full relative path from ./external_configs/
 * @property {boolean} is_dir - True if it's a directory
 * @property {ConfigNodeData[] | null} [children] - Nested children
 */

// --- Helper Functions ---

/**
 * Adapts API data structure to the structure expected by react-arborist.
 * @param {Array<object>} apiData - Data from the API (/configs/structure)
 * @returns {ConfigNodeData[]}
 */
const adaptApiDataToNodeData = (apiData) => {
    if (!Array.isArray(apiData)) return [];
    return apiData.map(item => ({
        id: item.path,
        name: item.path.split('/').pop() || item.path,
        relativePath: item.path,
        is_dir: item.is_dir,
        // Important: Only add children array if it's a directory. Leaf nodes should not have 'children'.
        children: item.is_dir ? (item.children ? adaptApiDataToNodeData(item.children) : null) : undefined,
    }));
};

// --- Node Renderer Component ---

/** @param {import("react-arborist").NodeRendererProps<ConfigNodeData> & { navigate: Function, onNodeClick: Function, lastClick: { id: string | null, time: number }, newlyCreatedItemId: string | null } } props */
function Node({ node, style, dragHandle, navigate, onNodeClick, lastClick, newlyCreatedItemId }) {
  const Icon = node.isLeaf ? FiFile : FiFolder;
  const iconColor = node.isLeaf ? 'text-blue-400' : 'text-yellow-500';
  const isNewlyCreated = node.id === newlyCreatedItemId;

  const handleRowClick = (e) => {
    const now = Date.now();
    const DOUBLE_CLICK_THRESHOLD = 300; // ms

    if (lastClick.id === node.id && (now - lastClick.time) < DOUBLE_CLICK_THRESHOLD) {
      // Double click detected
      if (node.isLeaf) {
        navigate(`/configs/${encodeURIComponent(node.data.relativePath)}`);
      } else if (node.isInternal) { // Handle folder double-click
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
    e.stopPropagation(); // Prevent row click logic if caret is clicked
    if (node.isInternal) {
      node.toggle(); // Toggle folder open/closed
    }
  };

  // Content structure: Caret (for folders), Icon, Name
  const nodeContent = (
    <div
      ref={dragHandle} // Files and folders can be dragged
      style={style}
      className={`flex items-center gap-2 px-2 py-1 rounded cursor-pointer
                  ${node.isSelected ? 'bg-blue-700' : 'hover:bg-gray-700'}
                  ${node.isEditing ? 'bg-gray-600' : ''}
                  ${isNewlyCreated ? 'animate-pulse-white' : ''} /* Apply pulse animation */
                 `}
      onClick={handleRowClick}
    >
      {/* Caret */}
      <span
        className="w-6 text-center flex items-center justify-center"
        onClick={handleCaretClick}
      >
        {/* Show caret only for folders (internal nodes) */}
        {node.isInternal && (node.isOpen ? <FiChevronDown /> : <FiChevronRight />)}
      </span>

      {/* Icon + Name */}
      <Icon className={`${iconColor} flex-shrink-0 mr-1`} />
      <span className="truncate flex-grow text-gray-200">
        {node.data.name}
      </span>
    </div>
  );

  // Return the div directly, no Link wrapper needed anymore
  return nodeContent;
}

// --- Main ConfigSelector Component ---

function ConfigSelector() {
  /** @type {[ConfigNodeData[] | null, Function]} */
  const [treeData, setTreeData] = useState(null); // Start as null to indicate initial loading
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  /** @type {[ConfigNodeData[], Function]} */
  const [selectedNodes, setSelectedNodes] = useState([]);
  const treeRef = useRef(null);
  const navigate = useNavigate(); // Get navigate function

  // State for manual double-click tracking
  const [lastClick, setLastClick] = useState({ id: null, time: 0 });
  // State for newly created item highlight
  const [newlyCreatedItemId, setNewlyCreatedItemId] = useState(null);

  const handleNodeClick = (id, time) => {
    setLastClick({ id, time });
    setNewlyCreatedItemId(null); // Clear highlight on any node click
  };

  // Modals State
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isCreateFolderModalOpen, setIsCreateFolderModalOpen] = useState(false);
  const [isDuplicateModalOpen, setIsDuplicateModalOpen] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const [duplicateSourceAlias, setDuplicateSourceAlias] = useState('');
  const [duplicateDestPath, setDuplicateDestPath] = useState('');
  const [availableAliases, setAvailableAliases] = useState([]);
  const [aliasSuggestions, setAliasSuggestions] = useState([]);
  const [isFetchingAliases, setIsFetchingAliases] = useState(false);
  const [modalError, setModalError] = useState(null);
  const [isSubmittingModal, setIsSubmittingModal] = useState(false);
  const aliasInputRef = useRef(null);

  // Action Status State
  const [isMoving, setIsMoving] = useState(false);
  const [moveError, setMoveError] = useState(null);
  const [actionStatus, setActionStatus] = useState({ message: '', error: false, success: false }); // For delete/create/duplicate feedback

  // State for shake animation trigger
  const [isShaking, setIsShaking] = useState(false); // RE-ADD state for shaking

  // State to track if alias details section is open in the duplicate modal
  const [isAliasDetailsOpen, setIsAliasDetailsOpen] = useState(false);

  // Refs for modal inputs
  const createFolderInputRef = useRef(null);

  // --- Data Fetching ---
  const fetchInitialData = useCallback(async () => {
    // Only set loading true on initial load (treeData is null)
    if (treeData === null) setIsLoading(true);
    setError(null);
    setMoveError(null);
    setActionStatus({ message: '', error: false, success: false }); // Clear status on refresh

    try {
      const structure = await fetchConfigStructure();
      let adaptedData = adaptApiDataToNodeData(structure);

      // Sort top-level items (folders first, then alphabetically)
      adaptedData.sort((a, b) => {
        if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1;
        return a.name.localeCompare(b.name);
      });
      setTreeData(adaptedData);
    } catch (e) {
      console.error("Failed to fetch config structure:", e);
      setError(e.message || 'Failed to load configurations. Is the API server running?');
      setTreeData([]); // Set to empty array on error to stop loading spinner
    } finally {
      setIsLoading(false);
    }
  }, [treeData]); // Depend on treeData to re-run only if it's null

  useEffect(() => {
    // Fetch data only on initial mount when treeData is null
    if (treeData === null) {
      fetchInitialData();
    }
  }, [fetchInitialData, treeData]);

  // Effect to handle newly created items: expand parents and scroll into view
  useEffect(() => {
    if (newlyCreatedItemId && treeRef.current && treeData) {
        const node = treeRef.current.get(newlyCreatedItemId);
        if (node) {
            // Expand all parents
            let parent = node.parent;
            while (parent && !parent.isOpen) {
                // Check if parent.id exists before trying to open
                if (parent.id) {
                    treeRef.current.open(parent.id, { autoSelect: false });
                } else {
                     // If the parent is the root (which might not have an id in some setups)
                     // or if there's an issue, break the loop.
                    console.warn("Could not open parent, ID missing:", parent);
                    break;
                }
                parent = parent.parent; // Move up the tree
            }

            // Scroll the new node into view after a short delay to allow rendering
            // It seems react-arborist might handle scrolling automatically on open,
            // but this ensures it if needed, especially after multiple opens.
            const timerId = setTimeout(() => {
                treeRef.current?.scrollTo(newlyCreatedItemId, "center"); // or "nearest", "start", "end"
            }, 100); // Small delay

            return () => clearTimeout(timerId); // Cleanup timeout on unmount or change
        } else {
            console.warn(`Node with ID ${newlyCreatedItemId} not found in tree after creation.`);
            // Optionally reset the ID if the node isn't found after a refresh
            // setNewlyCreatedItemId(null);
        }
    }
  }, [newlyCreatedItemId, treeData]); // Rerun when ID changes or tree data updates

  // Effect to trigger shake animation on selection change - RE-ADD
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

  // --- Alias Fetching for Duplicate Modal ---
  const fetchAliases = useCallback(async () => {
    if (isFetchingAliases) return;
    setIsFetchingAliases(true);
    setModalError(null); // Clear previous errors specific to alias fetching
    try {
      const aliases = await fetchAvailableConfigAliases(); // Assumes this API function exists
      setAvailableAliases(Array.isArray(aliases) ? aliases : []);
    } catch (e) {
      console.error("Failed to fetch config aliases:", e);
      setModalError("Could not load config aliases for duplication.");
      setAvailableAliases([]);
    } finally {
      setIsFetchingAliases(false);
    }
  }, [isFetchingAliases]);

  // --- Event Handlers ---

  const handleSelect = (nodes) => {
    setSelectedNodes(nodes.map(n => n.data));
    if (nodes.length > 0) {
        setNewlyCreatedItemId(null); // Clear highlight on selection change
    }
  };

  const handleToggle = (id) => {
    // Optional: Add logging or specific actions on toggle if needed
    // console.log('Toggled:', id);
    setNewlyCreatedItemId(null); // Clear highlight on toggle
  };

  /** @param {import("react-arborist").MoveHandler<ConfigNodeData>} args */
  const handleMove = async ({ dragIds, parentId, index }) => {
    setIsMoving(true);
    setMoveError(null);
    setActionStatus({ message: '', error: false, success: false });

    const parentNode = parentId ? treeRef.current?.get(parentId) : null;
    const parentPath = parentNode ? parentNode.data.relativePath : '.'; // Root is '.'

    // Prevent moving items into a file
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
        // Construct destination path: parentPath/itemName
        const cleanParentPath = parentPath === '.' ? '' : parentPath.replace(/^\/+|\/+$/g, '');
        const cleanItemName = itemName.replace(/^\/+/,'');
        const destinationPath = cleanParentPath ? `${cleanParentPath}/${cleanItemName}` : cleanItemName;

        console.log(`Attempting move: ${sourcePath} -> ${destinationPath}`);
        return moveConfigItem(sourcePath, destinationPath); // Use moveConfigItem
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
        // Clear error after delay
        setTimeout(() => setMoveError(null), 8000);
    } else {
      console.log(`Successfully moved ${successCount} item(s).`);
      // Optionally show success message in status bar
    }

    // Refresh tree if any move succeeded or failed to reflect server state
    if (successCount > 0 || firstError) {
      await fetchInitialData();
      setSelectedNodes([]); // Clear selection after move
      treeRef.current?.deselectAll();
    }
    setNewlyCreatedItemId(null); // Clear highlight on move
  };

  const handleDelete = async () => {
    if (selectedNodes.length === 0 || isSubmittingModal) return;

    const itemsToDelete = [...selectedNodes];
    setIsSubmittingModal(true);
    setModalError(null);
    setActionStatus({ message: '', error: false, success: false }); // Clear general status
    let successCount = 0;
    let firstError = null;

    const results = await Promise.allSettled(
      itemsToDelete.map(node => {
          // Call deleteConfigItem for both files and folders
          return deleteConfigItem(node.relativePath);
      })
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
      // Display error within the modal
      setModalError(`Error: ${firstError}${successCount > 0 ? ` (${successCount} file(s) deleted successfully)` : ''}`);
    } else {
      setActionStatus({ message: `Successfully deleted ${successCount} file(s).`, error: false, success: true });
      setIsDeleteModalOpen(false); // Close modal on full success
      setTimeout(() => setActionStatus({ message: '', error: false, success: false }), 5000); // Clear status message
    }

    // Refresh tree if any deletions succeeded
    if (successCount > 0) {
      await fetchInitialData();
      setSelectedNodes([]); // Clear selection
      treeRef.current?.deselectAll(); // Also deselect in the tree instance
    }
  };

  const handleCreateFolder = async () => {
    if (!newFolderName.trim() || isSubmittingModal) return;

    // Determine parent path based on selection (similar to InputManager)
    let parentPath = '.';
    let parentNode = null; // Keep track of the parent node
    const selectedDir = selectedNodes.find(n => n.is_dir);
    if (selectedNodes.length === 1 && selectedDir) {
      parentPath = selectedDir.relativePath;
      parentNode = treeRef.current?.get(selectedDir.id); // Get the node itself
    } else if (selectedNodes.length > 0) {
      const firstNodeId = selectedNodes[0].id;
      const firstNode = treeRef.current?.get(firstNodeId);
      const parent = firstNode?.parent;
      if (parent) {
          parentPath = parent.data.relativePath;
          parentNode = parent; // Get the parent node
      } else {
          parentPath = '.'; // Default to root if no parent found
          parentNode = null;
      }
    }

    // --- START EDIT: Attempt to open the parent directory ---
    if (parentNode && parentNode.id && !parentNode.isOpen && treeRef.current) {
        try {
            console.log(`LOG: Config.handleCreateFolder attempting to open parent node: ${parentNode.id}`);
            await treeRef.current.open(parentNode.id, { autoSelect: false });
            console.log(`LOG: Config.handleCreateFolder successfully opened parent node: ${parentNode.id}`);
        } catch (openError) {
             console.warn(`LOG: Config.handleCreateFolder failed to open parent node ${parentNode.id}:`, openError);
             // Continue anyway
        }
    }
    // --- END EDIT ---

    const cleanParentPath = parentPath === '.' ? '' : parentPath.replace(/^\/+|\/+$/g, '');
    const cleanNewFolderName = newFolderName.trim().replace(/^\/+|\/+$/g, '');
    const fullPath = cleanParentPath ? `${cleanParentPath}/${cleanNewFolderName}` : cleanNewFolderName;

    setIsSubmittingModal(true);
    setModalError(null);
    setActionStatus({ message: '', error: false, success: false });

    try {
      await createConfigDirectory(fullPath); // Use createConfigDirectory
      setNewFolderName('');
      setIsCreateFolderModalOpen(false);
      setActionStatus({ message: `Successfully created folder '${cleanNewFolderName}'.`, error: false, success: true });
      setTimeout(() => setActionStatus({ message: '', error: false, success: false }), 5000);
      // Set the newly created ID *before* fetching data
      setNewlyCreatedItemId(fullPath);
      await fetchInitialData(); // Refresh tree
    } catch (e) {
      console.error("Failed to create directory:", e);
      setModalError(e.message || 'Could not create folder.'); // Show error in modal
    } finally {
      setIsSubmittingModal(false);
    }
  };

  const handleDuplicate = async () => {
      if (!duplicateSourceAlias.trim() || !duplicateDestPath.trim() || isSubmittingModal) return;

      const destPathClean = duplicateDestPath.trim().replace(/^\/+|\/+$/g, '');
      const destPathParts = destPathClean.split('/');
      let destParentPath = '.';
      let destParentNode = null;

      // Determine the parent path and node for the destination
      if (destPathParts.length > 1) {
          destParentPath = destPathParts.slice(0, -1).join('/');
          destParentNode = treeRef.current?.get(destParentPath);
      } // else it's in the root, parentNode remains null

      // --- START EDIT: Attempt to open the destination parent directory ---
      if (destParentNode && destParentNode.id && !destParentNode.isOpen && treeRef.current) {
          try {
              console.log(`LOG: Config.handleDuplicate attempting to open dest parent node: ${destParentNode.id}`);
              await treeRef.current.open(destParentNode.id, { autoSelect: false });
              console.log(`LOG: Config.handleDuplicate successfully opened dest parent node: ${destParentNode.id}`);
          } catch (openError) {
               console.warn(`LOG: Config.handleDuplicate failed to open dest parent node ${destParentNode.id}:`, openError);
               // Continue duplication anyway
          }
      }
      // --- END EDIT ---

      setIsSubmittingModal(true);
      setModalError(null);
      setActionStatus({ message: '', error: false, success: false });

      try {
          await duplicateConfig(duplicateSourceAlias, destPathClean);
          setDuplicateSourceAlias('');
          setDuplicateDestPath('');
          setAliasSuggestions([]);
          setIsDuplicateModalOpen(false);
          setActionStatus({ message: `Successfully duplicated '${duplicateSourceAlias}' to '${destPathClean}'.`, error: false, success: true });
          setTimeout(() => setActionStatus({ message: '', error: false, success: false }), 5000);
           // Set the newly created ID *before* fetching data
           setNewlyCreatedItemId(destPathClean);
          await fetchInitialData(); // Refresh tree
      } catch (e) {
          console.error("Failed to duplicate config:", e);
          setModalError(e.message || 'Could not duplicate config.'); // Show error in modal
      } finally {
          setIsSubmittingModal(false);
      }
  };


  // --- Modal Opening/Closing ---

  const openCreateFolderModal = () => {
    setNewFolderName('');
    setModalError(null);
    setIsCreateFolderModalOpen(true);
    setNewlyCreatedItemId(null); // Clear highlight when opening modal
    // Autofocus the input field
    setTimeout(() => createFolderInputRef.current?.focus(), 100);
  }

  const openDeleteModal = () => {
    setModalError(null);
    setIsDeleteModalOpen(true);
    setNewlyCreatedItemId(null); // Clear highlight when opening modal
  }

  const openDuplicateModal = () => {
    setDuplicateSourceAlias('');
    setDuplicateDestPath('');
    setModalError(null);
    setAliasSuggestions([]);
    setIsAliasDetailsOpen(false); // Reset details open state when modal opens
    setIsDuplicateModalOpen(true);
    setNewlyCreatedItemId(null); // Clear highlight when opening modal
    // Fetch aliases when the modal opens, if not already fetched
    if (availableAliases.length === 0) {
        fetchAliases();
    }
    // Focus input after a short delay
    setTimeout(() => aliasInputRef.current?.focus(), 100);
  }

  // --- Alias Suggestion Logic ---
  const handleAliasInputChange = (e) => {
    const value = e.target.value;
    setDuplicateSourceAlias(value);
    if (value && availableAliases.length > 0) {
        const lowerValue = value.toLowerCase();
        setAliasSuggestions(
            availableAliases.filter(a => a.toLowerCase().includes(lowerValue)).slice(0, 5) // Limit suggestions
        );
    } else {
        setAliasSuggestions([]);
    }
  };

  const handleAliasSuggestionClick = (suggestion) => {
      setDuplicateSourceAlias(suggestion);
      setAliasSuggestions([]);
      aliasInputRef.current?.focus();
  };

  // --- Utilities ---

  // Function to clear selection when clicking outside the tree or action buttons
  const handleBackgroundClick = useCallback((e) => {
      // Don't clear if a modal is open
      if (isDeleteModalOpen || isCreateFolderModalOpen || isDuplicateModalOpen) return;

      const clickedOnRow = e.target.closest('[role="treeitem"]');
      const clickedOnScrollbar = e.target.closest('.simplebar-scrollbar'); // Adjust if using a specific scrollbar library
      const clickedOnActionButton = e.target.closest('[data-action-button="true"]');
      const clickedOnModal = e.target.closest('[role="dialog"]'); // Check if click is inside any modal
      const clickedInsideDetails = e.target.closest('details'); // Don't clear if clicking inside the alias details

      if (!clickedOnRow && !clickedOnScrollbar && !clickedOnActionButton && !clickedOnModal && !clickedInsideDetails) {
          if (treeRef.current?.deselectAll) {
              treeRef.current.deselectAll();
          }
          if (typeof setSelectedNodes === 'function') {
              setSelectedNodes([]);
          }
           // Also clear the highlight on background click
           setNewlyCreatedItemId(null);
      }
  }, [isDeleteModalOpen, isCreateFolderModalOpen, isDuplicateModalOpen]); // Dependencies to re-create handler if modal state changes


  const actionButtonClass = "flex items-center gap-2 px-3 py-1.5 rounded bg-gray-700 text-gray-200 text-sm transition-colors duration-200 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed";

  // Define scroll speeds
  const slowScrollDuration = 30; // seconds
  const fastScrollDuration = 10; // seconds

  // Determine current scroll duration based on selection
  const currentScrollDuration = selectedNodes.length > 0 ? fastScrollDuration : slowScrollDuration;

  // --- Render ---
  return (
    <div className="flex flex-col h-full relative" onMouseDown={handleBackgroundClick}>
      {/* Animated Background - pass dynamic duration */}
      <ScrollingTextBackground duration={currentScrollDuration} />

      {/* Main Content Area (Wrapped and Centered) */}
      <div className="relative z-10 flex-grow flex flex-col max-w-7xl w-full mx-auto p-6">

        {/* Header */}
        <div className="flex justify-between items-center mb-4 bg-gray-900 bg-opacity-80 backdrop-blur-sm p-4 rounded-t-lg border-b border-gray-700">
          <span>
            <h1 className="text-2xl font-bold text-gray-100">Configuration Manager</h1>
            <h2 className="text-lg text-gray-300 mt-1 animate-pulse-text-subtle">Select a config file to get started</h2>
          </span>
          <div className="flex items-center gap-2">
            {/* EnhancedButton for Create Folder */}
            <EnhancedButton
              onClick={openCreateFolderModal}
              className={`${actionButtonClass} hover:bg-green-600 ${selectedNodes.length > 0 ? 'opacity-75' : ''}`} // Keep existing hover for color change
              title="Create new folder in selected location or root"
              data-action-button="true"
              particleColor="#22c55e" // Green particles
            >
              <FiFolderPlus /> Create Folder
            </EnhancedButton>
            {/* EnhancedButton for Duplicate Config */}
            <EnhancedButton
              onClick={openDuplicateModal}
              className={`${actionButtonClass} hover:bg-blue-600 ${selectedNodes.length > 0 ? 'opacity-75' : ''}`} // Keep existing hover for color change
              title="Duplicate a config from a source alias"
              data-action-button="true"
              particleColor="#3b82f6" // Blue particles
            >
              <FiCopy /> Duplicate Config
            </EnhancedButton>
            {/* Container for Delete button and hint - positioned relatively */}
            <div className="relative">
              {/* Conditionally show "double click" hint - positioned absolutely */}
              {selectedNodes.length === 1 && !selectedNodes[0].is_dir && (
                <p className="absolute bottom-full right-0 mb-1 text-xs italic text-gray-200 whitespace-nowrap">
                  or double click to open
                </p>
              )}
              {/* EnhancedButton for Delete */}
              <EnhancedButton
                onClick={openDeleteModal}
                className={`${actionButtonClass} ${selectedNodes.length > 0
                    ? 'animate-pulse-red' // Keep existing pulse when selected
                    : 'text-red-400 hover:bg-red-900 hover:text-red-300' // Default styles
                    } ${isShaking ? 'animate-shake' : ''}`} // RE-ADD shake animation class
                title="Delete Selected File(s) and/or Folder(s)"
                disabled={selectedNodes.length === 0} // Still disabled when nothing selected
                data-action-button="true"
                particleColor="#ef4444" // Red particles
              >
                <FiTrash2 /> Delete
              </EnhancedButton>
            </div>
          </div>
        </div>

        {/* Loading/Error States for Initial Load - Apply background */}
        {isLoading && (
          <div className="flex-grow flex items-center justify-center text-gray-400 bg-gray-900 bg-opacity-80 backdrop-blur-sm p-4 rounded-b-lg">
            <FiLoader className="animate-spin mr-3" /> Loading configurations...
          </div>
        )}
        {!isLoading && error && (
          <div className="flex-grow flex flex-col items-center justify-center text-red-400 bg-red-900 bg-opacity-60 backdrop-blur-sm p-6 rounded-b-lg border border-red-700">
            <FiAlertTriangle className="w-10 h-10 mb-2" />
            <p>Error loading configuration structure:</p>
            <p className="text-sm mt-1">{error}</p>
            {/* EnhancedButton for Retry */}
            <EnhancedButton onClick={fetchInitialData} className="mt-4 px-3 py-1.5 rounded bg-gray-600 hover:bg-gray-500 text-gray-200 text-sm" particleColor="#9ca3af">
              Retry
            </EnhancedButton>
          </div>
        )}

        {/* Tree View Area - Apply background */}
        {!isLoading && !error && (
          <div className="flex-grow border-b border-r border-l border-gray-700 rounded-b-lg bg-gray-900 bg-opacity-80 backdrop-blur-sm overflow-auto relative min-h-0"> {/* Ensure flex-grow and min-height */} 
            {(treeData && treeData.length > 0) ? (
              <Tree
                ref={treeRef}
                data={treeData}
                openByDefault={false}
                width="100%"
                height={1000} // Reverted to a large number, container handles scrolling
                indent={24}
                rowHeight={32}
                paddingTop={10}
                paddingBottom={10}
                onSelect={handleSelect}
                onToggle={handleToggle}
                onMove={handleMove}
                disableDrop={({ parentNode, dropTarget }) => {
                  // Prevent dropping *onto* a file
                  return dropTarget && !dropTarget.data.is_dir;
                }}
                // Selection follows focus can be useful for multi-select + drag
                // selectionFollowsFocus={true}
              >
                {/* Pass props needed by Node, including click tracking and newly created ID */}
                {(props) => <Node {...props} navigate={navigate} onNodeClick={handleNodeClick} lastClick={lastClick} newlyCreatedItemId={newlyCreatedItemId} />}
              </Tree>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-gray-500">
                <p>No configuration files or folders found in ./external_configs/</p>
                {/* Optional: Add buttons to create folder/duplicate even when empty */}
              </div>
            )}
          </div>
        )}

        {/* Status Bar Area - Apply background */}
        <div className="h-8 mt-2 text-sm flex items-center gap-4 overflow-hidden whitespace-nowrap px-4 py-1 bg-gray-900 bg-opacity-80 backdrop-blur-sm rounded-lg border border-gray-700">
          {/* Moving Status */}
          {isMoving && (
              <div className="flex items-center text-blue-400">
                  <FiLoader className="animate-spin mr-2 flex-shrink-0" /> Moving items...
              </div>
          )}
          {moveError && <p className="text-red-400 truncate" title={moveError}>Move Error: {moveError}</p>}

          {/* General Action Status (Delete, Create, Duplicate) */}
          {actionStatus.message && (
              <p className={`${actionStatus.error ? 'text-red-400' : actionStatus.success ? 'text-green-400' : 'text-gray-400'} truncate flex items-center`} title={actionStatus.message}>
                  {actionStatus.success && <FiCheckCircle className="mr-1 flex-shrink-0"/>}
                  {actionStatus.error && <FiAlertTriangle className="mr-1 flex-shrink-0"/>}
                  {actionStatus.message}
              </p>
          )}
        </div>

      </div> {/* End Main Content Wrapper */}


      {/* --- Modals --- */}
      {/* Modals remain outside the centered content, possibly covering the whole screen */}

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={isDeleteModalOpen}
        onClose={() => !isSubmittingModal && setIsDeleteModalOpen(false)}
        title="Confirm Deletion"
      >
        <p className="mb-4">Are you sure you want to delete the following {selectedNodes.length} item(s) (files and folders)? This action cannot be undone.</p>
        <ul className="list-disc list-inside mb-4 max-h-40 overflow-y-auto bg-gray-700 p-2 rounded border border-gray-600">
          {/* Show all selected items, files and folders */}
          {selectedNodes.map(node => <li key={node.id} className="truncate text-sm py-0.5 flex items-center gap-1" title={node.relativePath}>
              {node.is_dir ? <FiFolder className="inline-block w-4 h-4 text-yellow-500 flex-shrink-0" /> : <FiFile className="inline-block w-4 h-4 text-blue-400 flex-shrink-0" />}
              <span>{node.relativePath}</span>
            </li>)}
        </ul>
        {modalError && (
          <p className="text-red-400 text-sm mb-3 bg-red-900 bg-opacity-40 p-2 rounded border border-red-700">{modalError}</p>
        )}
        <div className="flex justify-end gap-3 mt-4">
          {/* Standard button for Cancel */}
          <button
            type="button"
            onClick={() => setIsDeleteModalOpen(false)}
            className="px-4 py-2 rounded bg-gray-600 hover:bg-gray-500 text-gray-200 transition-colors disabled:opacity-50"
            disabled={isSubmittingModal}
          >
            Cancel
          </button>
          {/* EnhancedButton for Confirm Delete */}
          <EnhancedButton
            onClick={handleDelete}
            className={`px-4 py-2 rounded bg-red-600 hover:bg-red-500 text-white flex items-center disabled:opacity-50
                       ${isDeleteModalOpen && !isSubmittingModal && selectedNodes.length > 0 ? 'animate-pulse-red' : ''}`}
            disabled={isSubmittingModal || selectedNodes.length === 0} // Disable only if nothing selected or submitting
            particleColor="#f87171" // Light red particles to contrast with bg
          >
            {isSubmittingModal ? <FiLoader className="animate-spin mr-2" /> : <FiTrash2 className="mr-1" />}
            {isSubmittingModal ? 'Deleting...' : `Delete ${selectedNodes.length} Item(s)`}
          </EnhancedButton>
        </div>
      </Modal>

      {/* Create Folder Modal */}
      <Modal
        isOpen={isCreateFolderModalOpen}
        onClose={() => !isSubmittingModal && setIsCreateFolderModalOpen(false)}
        title="Create New Folder"
      >
        <label htmlFor="newFolderName" className="block mb-2 text-sm font-medium text-gray-300">Folder Name:</label>
        <input
          ref={createFolderInputRef}
          id="newFolderName"
          type="text"
          value={newFolderName}
          onChange={(e) => setNewFolderName(e.target.value)}
          placeholder="e.g., my_configs or nested/folder"
          className={`w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 mb-1
                      ${!newFolderName.trim() && isCreateFolderModalOpen ? 'animate-pulse-bg-subtle' : ''}`}
          disabled={isSubmittingModal}
          onKeyDown={(e) => e.key === 'Enter' && !isSubmittingModal && newFolderName.trim() && handleCreateFolder()}
        />
        <p className="text-xs text-gray-400 mb-3">Folder will be created inside the selected folder, or at the root if none is selected.</p>
        {modalError && <p className="text-red-400 text-xs mt-1 mb-3">Error: {modalError}</p>}
        <div className="flex justify-end gap-3 mt-4">
          {/* Standard button for Cancel */}
          <button
            type="button"
            onClick={() => setIsCreateFolderModalOpen(false)}
            className="px-4 py-2 rounded bg-gray-600 hover:bg-gray-500 text-gray-200 transition-colors disabled:opacity-50"
            disabled={isSubmittingModal}
          >
            Cancel
          </button>
          {/* EnhancedButton for Confirm Create */}
          <EnhancedButton
            onClick={handleCreateFolder}
            className={`px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white flex items-center disabled:opacity-50
                       ${newFolderName.trim() && !isSubmittingModal ? 'animate-pulse-blue-gold' : ''}`}
            disabled={!newFolderName.trim() || isSubmittingModal}
            particleColor="#60a5fa" // Light blue particles
          >
            {isSubmittingModal ? <FiLoader className="animate-spin mr-2" /> : <FiFolderPlus className="mr-1" />}
            {isSubmittingModal ? 'Creating...' : 'Create'}
          </EnhancedButton>
        </div>
      </Modal>

        {/* Duplicate Config Modal */}
        <Modal
            isOpen={isDuplicateModalOpen}
            onClose={() => !isSubmittingModal && setIsDuplicateModalOpen(false)}
            title="Duplicate Configuration"
        >
            {/* Source Alias Input */}
            <div className="mb-4 relative">
                <label htmlFor="duplicateSourceAlias" className="block mb-1 text-sm font-medium text-gray-300">Source Alias:</label>
                <input
                    ref={aliasInputRef}
                    id="duplicateSourceAlias"
                    type="text"
                    value={duplicateSourceAlias}
                    onChange={handleAliasInputChange}
                    placeholder="Type to search or select from list below..."
                    className={`w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                                 ${!duplicateSourceAlias.trim() && !isAliasDetailsOpen && isDuplicateModalOpen ? 'animate-pulse-bg-subtle' : ''}`}
                    disabled={isSubmittingModal || isFetchingAliases}
                />
                {isFetchingAliases && <FiLoader className="animate-spin absolute right-3 top-9 transform text-gray-400"/>}
                {/* Suggestions Dropdown */}
                {aliasSuggestions.length > 0 && (
                    <ul className="absolute z-10 w-full mt-1 bg-gray-700 border border-gray-600 rounded shadow-lg max-h-40 overflow-y-auto">
                        {aliasSuggestions.map((alias) => (
                            <li
                                key={alias}
                                className="px-3 py-2 cursor-pointer hover:bg-gray-600 text-gray-200 text-sm"
                                onClick={() => handleAliasSuggestionClick(alias)}
                            >
                                {alias}
                            </li>
                        ))}
                    </ul>
                )}
                {!isFetchingAliases && availableAliases.length === 0 && !modalError && !isFetchingAliases && (
                     <p className="text-xs text-yellow-400 mt-1">No config aliases found or failed to load.</p>
                )}
            </div>

            {/* Expandable List of All Aliases */}
            {!isFetchingAliases && availableAliases.length > 0 && (
                <details
                    className={`mb-4 rounded border border-gray-600 overflow-hidden
                               ${isAliasDetailsOpen && !duplicateSourceAlias.trim() ? 'animate-pulse-border-subtle' : ''}`}
                    onToggle={(e) => setIsAliasDetailsOpen(e.target.open)} // Track open state
                >
                    <summary className="px-3 py-2 cursor-pointer bg-gray-700 hover:bg-gray-650 text-sm text-gray-300 flex justify-between items-center">
                        <span>Show All Available Aliases ({availableAliases.length})</span>
                        {/* Simple triangle indicator */}
                        <span className="details-marker">▼</span>
                    </summary>
                    <ul className="bg-gray-800 max-h-48 overflow-y-auto p-2">
                        {availableAliases
                            .slice() // Create a copy to avoid modifying the original state
                            .sort((a, b) => a.localeCompare(b)) // Sort alphabetically
                            .map((alias) => (
                            <li
                                key={alias}
                                className="px-3 py-1.5 cursor-pointer hover:bg-gray-700 text-gray-200 text-sm rounded"
                                onClick={(e) => {
                                    handleAliasSuggestionClick(alias);
                                    // Optionally close the details element after selection
                                    e.target.closest('details')?.removeAttribute('open');
                                }}
                            >
                                {alias}
                            </li>
                        ))}
                    </ul>
                </details>
            )}
            {/* CSS for the details marker - could be added to a global CSS file instead */}
            <style>{`
                details > summary { list-style: none; }
                details > summary::-webkit-details-marker { display: none; }
                details > summary .details-marker { transition: transform 0.2s; }
                details[open] > summary .details-marker { transform: rotate(180deg); }
            `}</style>


            {/* Destination Path Input */}
            <div className="mb-4">
                <label htmlFor="duplicateDestPath" className="block mb-1 text-sm font-medium text-gray-300">Destination Path:</label>
                <input
                    id="duplicateDestPath"
                    type="text"
                    value={duplicateDestPath}
                    onChange={(e) => setDuplicateDestPath(e.target.value)}
                    placeholder="e.g., new_config.yaml or folder/new_config.yaml"
                    className={`w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                                 ${duplicateSourceAlias.trim() && !duplicateDestPath.trim() && isDuplicateModalOpen ? 'animate-pulse-bg-subtle' : ''}`}
                    disabled={isSubmittingModal}
                />
                <p className="text-xs text-gray-400 mt-1">Path relative to <code className="text-xs bg-gray-600 px-1 rounded">./external_configs/</code>. Must end in
                    <code className={`text-xs bg-gray-600 px-1 rounded ml-1
                                     ${duplicateDestPath.trim() && !duplicateDestPath.trim().toLowerCase().endsWith('.yaml') ? 'text-red-400 font-bold' : ''}
                                     `}>
                        .yaml
                    </code>.
                </p>
            </div>

            {/* Modal Error Display */}
            {modalError && <p className="text-red-400 text-sm mb-3 bg-red-900 bg-opacity-40 p-2 rounded border border-red-700">{modalError}</p>}

            {/* Modal Actions */}
            <div className="flex justify-end gap-3 mt-4">
                {/* Standard button for Cancel */}
                <button
                    type="button"
                    onClick={() => setIsDuplicateModalOpen(false)}
                    className="px-4 py-2 rounded bg-gray-600 hover:bg-gray-500 text-gray-200 transition-colors disabled:opacity-50"
                    disabled={isSubmittingModal}
                >
                    Cancel
                </button>
                {/* EnhancedButton for Confirm Duplicate */}
                <EnhancedButton
                    onClick={handleDuplicate}
                    className={`px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white flex items-center disabled:opacity-50
                               ${duplicateSourceAlias.trim() && duplicateDestPath.trim().toLowerCase().endsWith('.yaml') && !isSubmittingModal ? 'animate-pulse-blue-gold' : ''}
                               `}
                    disabled={!duplicateSourceAlias.trim() || !duplicateDestPath.trim().toLowerCase().endsWith('.yaml') || isSubmittingModal || isFetchingAliases}
                    particleColor="#60a5fa" // Light blue particles
                >
                    {isSubmittingModal ? <FiLoader className="animate-spin mr-2" /> : <FiCopy className="mr-1" />}
                    {isSubmittingModal ? 'Duplicating...' : 'Duplicate'}
                </EnhancedButton>
            </div>
        </Modal>

    </div> // End main div
  );
}

export default ConfigSelector; 