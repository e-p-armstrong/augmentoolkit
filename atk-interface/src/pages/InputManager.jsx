import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Tree } from 'react-arborist';
import { FiFile, FiFolder, FiFolderPlus, FiUploadCloud, FiDownloadCloud, FiTrash2, FiLoader, FiAlertTriangle, FiCheckCircle, FiChevronDown, FiChevronRight } from 'react-icons/fi';
import { useDropzone } from 'react-dropzone';
import { uploadInputFiles, downloadInputItem, deleteInputItem, createInputDirectory, moveInputItem } from '../api';
import { useFileManager } from '../context/FileManagerContext';
import Modal from '../components/Modal';
import JSZip from 'jszip';
import PourOverBackground from '../components/PourOverBackground';
import EnhancedButton from '../components/EnhancedButton';

/**
 * @typedef {object} InputNodeData
 * @property {string} id - Unique ID (can be the relative path)
 * @property {string} name - Display name (basename of the path)
 * @property {boolean} [isInternal] - Flag for placeholder nodes
 * @property {string} relativePath - Store the full relative path
 * @property {boolean} is_dir - Keep the directory flag
 * @property {InputNodeData[] | null} [children] - Children array or null if leaf
 */

/** @param {import("react-arborist").NodeRendererProps<InputNodeData> & { onNodeClick: Function, lastClick: { id: string | null, time: number }, newlyCreatedItemId: string | null }} props */
function Node({ node, style, dragHandle, onNodeClick, lastClick, newlyCreatedItemId }) {
  const Icon = node.isLeaf ? FiFile : FiFolder;
  const iconColor = node.isLeaf ? 'text-blue-400' : 'text-yellow-500';
  const isNewlyCreated = node.id === newlyCreatedItemId;

  // Optional: add row‑click to select *and* open/close
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

  // Optional: only the caret toggles
  const handleCaretClick = (e) => {
    e.stopPropagation();       // keep the row‑click logic independent
    node.toggle();
  };

  return (
    <div
      ref={dragHandle}
      style={style}
      className={`flex items-center gap-2 px-2 py-1 rounded cursor-pointer
                  ${node.isSelected ? 'bg-blue-700' : 'hover:bg-gray-700'}
                  ${node.isEditing ? 'bg-gray-600' : ''}
                  ${isNewlyCreated ? 'animate-pulse-white' : ''}`}
      onClick={handleRowClick} // Use single click handler for selection/toggle
    >
      {/* caret */}
      <span
        className="w-6 text-center flex items-center justify-center"
        onClick={handleCaretClick}      /* <-- give it life */
      >
        {node.isInternal && (node.isOpen ? <FiChevronDown /> : <FiChevronRight />)}
      </span>

      {/* icon + name */}
      <Icon className={`${iconColor} flex-shrink-0 mr-1`} />
      <span 
        className="truncate flex-grow text-gray-200"
      >
        {node.data.name}
      </span>
    </div>
  );
}


function InputManager() {
    const { inputTreeData: treeData, isInputLoading: isLoading, inputError: error, fetchInputData, refreshInputData } = useFileManager();
    /** @type {[InputNodeData[], Function]} */
    const [selectedNodes, setSelectedNodes] = useState([]);
    const treeRef = useRef(null);
    console.log("InputManager RENDERED"); // Log component render
    // State for manual double-click tracking
    const [lastClick, setLastClick] = useState({ id: null, time: 0 });
    // State for newly created item highlight
    const [newlyCreatedItemId, setNewlyCreatedItemId] = useState(null);
    // State to store names of items from the last upload for highlighting
    const [lastUploadedNames, setLastUploadedNames] = useState([]);
    // State to trigger the search effect after an upload cycle completes
    const [uploadCycle, setUploadCycle] = useState(0);

    // Modals State
    const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
    const [isCreateFolderModalOpen, setIsCreateFolderModalOpen] = useState(false);
    const [newFolderName, setNewFolderName] = useState('');
    const [modalError, setModalError] = useState(null);
    const [isSubmittingModal, setIsSubmittingModal] = useState(false);

    // Upload Feedback State
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0); 
    const [uploadError, setUploadError] = useState(null);
    const [lastUploadResult, setLastUploadResult] = useState(null);

    // Download state
    const [isDownloading, setIsDownloading] = useState(false);
    const [downloadStatus, setDownloadStatus] = useState({ message: '', error: false });

    // Move/Drag State
    const [isMoving, setIsMoving] = useState(false);
    const [moveError, setMoveError] = useState(null);

    // State for shake animation trigger
    const [isShaking, setIsShaking] = useState(false);

    // --- State for PourOverBackground ---
    const [bgIsActive, setBgIsActive] = useState(false);
    const [triggerBgDownload, setTriggerBgDownload] = useState(false);
    const [triggerBgDelete, setTriggerBgDelete] = useState(false);

    // Refs for modal inputs
    const createFolderInputRef = useRef(null);

    useEffect(() => {
        fetchInputData(); // Fetch data on mount (uses cache if available)
    }, [fetchInputData]);

    // --- Handlers for PourOverBackground Callbacks ---
    const handleDownloadAnimComplete = useCallback(() => {
        setTriggerBgDownload(false);
        console.log("LOG: handleDownloadAnimComplete called");
    }, []);

    const handleDeleteAnimComplete = useCallback(() => {
        setTriggerBgDelete(false);
        console.log("LOG: handleDeleteAnimComplete called");
    }, []);

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

    // Effect to handle newly created items: expand parents and scroll into view
    useEffect(() => {
      console.log(`LOG: Highlight EFFECT triggered. newlyCreatedItemId: ${newlyCreatedItemId}, treeData available: ${!!treeData}`);
      if (newlyCreatedItemId && treeRef.current && treeData) {
          console.log(`LOG: Highlight EFFECT processing ID: ${newlyCreatedItemId}`);
          const node = treeRef.current.get(newlyCreatedItemId);
          console.log(`LOG: Highlight EFFECT node found: ${!!node}`);
          if (node) {
              // Expand all parents
              let parent = node.parent;
              console.log("LOG: Highlight EFFECT expanding parents...");
              while (parent && !parent.isOpen) {
                  console.log(`LOG: Highlight EFFECT opening parent: ${parent.id}`);
                  if (parent.id) {
                      treeRef.current.open(parent.id, { autoSelect: false });
                  } else {
                      console.warn("Could not open parent, ID missing:", parent);
                      break;
                  }
                  parent = parent.parent; // Move up the tree
              }
              // Scroll the new node into view
              const timerId = setTimeout(() => {
                  treeRef.current?.scrollTo(newlyCreatedItemId, "center");
                  console.log(`LOG: Highlight EFFECT scrolled to ${newlyCreatedItemId}`);
              }, 100);
              return () => clearTimeout(timerId);
          } else {
              console.warn(`Node with ID ${newlyCreatedItemId} not found in tree after creation/upload.`);
          }
      }
    }, [newlyCreatedItemId, treeData]); // Rerun when ID changes or tree data updates

    // Effect to find and highlight the first newly uploaded item after data refresh
    useEffect(() => {
        console.log(`LOG: Uploaded item search EFFECT triggered. uploadCycle: ${uploadCycle}, lastUploadedNames: [${lastUploadedNames.join(', ')}], treeData available: ${!!treeData}`);
        // Search the raw treeData, not just visible nodes
        if (lastUploadedNames.length > 0 && Array.isArray(treeData)) {
            let foundNodeId = null;
            console.log("LOG: Uploaded item search EFFECT searching raw treeData...");

            // Recursive function to search the raw data structure
            const findNodeInData = (nodes) => {
                if (!Array.isArray(nodes)) return null;

                for (const node of nodes) {
                    // node here is the raw data object (e.g., { id: 'path', name: 'name', ...})
                    // console.log(`LOG: Searching... Checking node: ${node.name} (ID: ${node.id})`);
                    if (node.isdir && node.children) {
                        console.log(`LOG: Searching... Checking directory node: ${node.name} (ID: ${node.id})`); // only log dir checks because there are a LOT of files here
                    }
                    if (lastUploadedNames.includes(node.name)) {
                        console.log(`LOG: Uploaded item search EFFECT found match: ${node.name} (ID: ${node.id})`);
                        return node.id; // Return the ID (path) of the first match
                    }
                    // If it's a directory, recurse into its children
                    if (node.is_dir && node.children) {
                        const foundInChildren = findNodeInData(node.children);
                        if (foundInChildren) {
                            return foundInChildren;
                        }
                    }
                }
                return null;
            };

            // Start search from the root of treeData
            foundNodeId = findNodeInData(treeData);

            if (foundNodeId) {
                console.log(`LOG: Uploaded item search EFFECT setting newlyCreatedItemId = ${foundNodeId}`);
                setNewlyCreatedItemId(foundNodeId);
                // Clear the names list ONLY after successfully finding and setting the highlight
                console.log("LOG: Uploaded item search EFFECT clearing lastUploadedNames after success.");
                setLastUploadedNames([]);
            } else {
                console.log("LOG: Uploaded item search EFFECT found no matching node in current treeData. Will retry if treeData updates.");
                // Optional: Could implement a deeper search if needed, but start simple
            }
        }
    }, [treeData, lastUploadedNames, uploadCycle]); // Depend on treeData, names, and the upload cycle trigger

    const handleUpload = useCallback(async (fileList, targetPath) => {
      const files = Array.from(fileList);
      console.log(`LOG: handleUpload START. Target: ${targetPath}, Files:`, files);

      // Check whether any item is a directory (webkitRelativePath !== '')
      const containsDir = files.some(f => f.webkitRelativePath);
      console.log(`LOG: handleUpload containsDir: ${containsDir}`);

      let payload = files;
      if (containsDir) {
        console.log("LOG: handleUpload zipping directories...");
        // zip each top‑level folder
        const groups = {};
        files.forEach(f => {
          const [root] = f.webkitRelativePath.split('/');
          (groups[root] ||= []).push(f);
        });
    
        const zips = await Promise.all(
          Object.entries(groups).map(async ([folder, fs]) => {
            const zip = new JSZip();
            fs.forEach(f =>
              zip.file(f.webkitRelativePath.replace(`${folder}/`, ''), f)
            );
            const blob = await zip.generateAsync({ type: 'blob' });
            return new File([blob], `${folder}.zip`, { type: 'application/zip' });
          })
        );
        payload = zips;
      }
    
      // Trigger background activity briefly on upload? (Optional)
      setBgIsActive(true);
      setTimeout(() => setBgIsActive(false), 1000); // Example: active for 1 sec

      console.log("LOG: handleUpload BEFORE uploadInputFiles call");
      await uploadInputFiles(targetPath, payload);
      console.log("LOG: handleUpload AFTER uploadInputFiles call");
      // Highlight is now set in onDrop *after* this completes
      // if (target !== '.') {
      //     setNewlyCreatedItemId(targetPath);
      // }
      console.log("LOG: handleUpload BEFORE refreshInputData call");
      refreshInputData(); // Refresh data after upload
      console.log("LOG: handleUpload AFTER refreshInputData call (refresh initiated)");
    }, [refreshInputData]);

    const handleSelect = (nodes) => {
        const selectedData = nodes.map(n => n.data);
        console.log("LOG: handleSelect called. Selected nodes count:", selectedData.length);
        setSelectedNodes(selectedData);
        // Set background to active if something is selected
        setBgIsActive(selectedData.length > 0);
        if (selectedData.length > 0) {
            console.log(`LOG: Clearing newlyCreatedItemId (${newlyCreatedItemId}) due to handleSelect.`);
            setNewlyCreatedItemId(null); // Clear highlight on selection
            console.log(`LOG: Clearing lastUploadedNames due to handleSelect.`);
            setLastUploadedNames([]); // Also clear search list
        }
    };

    const handleToggle = (id) => {
        console.log('LOG: handleToggle called. Toggled ID:', id);
        console.log(`LOG: Clearing newlyCreatedItemId (${newlyCreatedItemId}) due to handleToggle.`);
        setNewlyCreatedItemId(null); // Clear highlight on toggle
        console.log(`LOG: Clearing lastUploadedNames due to handleToggle.`);
        setLastUploadedNames([]); // Also clear search list
    };

    // --- Move Handler ---
    /** @param {import("react-arborist").MoveHandler<InputNodeData>} args */
    const handleMove = async ({ dragIds, parentId, index }) => {
        setIsMoving(true);
        setMoveError(null);
        console.log("Move detected:", { dragIds, parentId, index });

        const parentNode = parentId ? treeRef.current?.get(parentId) : null;
        const parentPath = parentNode ? parentNode.data.relativePath : '.'; // Root is '.'
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
                 // Construct destination path: parentPath/itemName
                 // Ensure no leading/trailing slashes on parentPath (unless it's root '.') 
                 // and no leading slash on itemName
                const cleanParentPath = parentPath === '.' ? '' : parentPath.replace(/^\/+|\/+$/g, '');
                const cleanItemName = itemName.replace(/^\/+/,'');
                const destinationPath = cleanParentPath ? `${cleanParentPath}/${cleanItemName}` : cleanItemName;

                console.log(`Attempting move: ${sourcePath} -> ${destinationPath}`);
                return moveInputItem(sourcePath, destinationPath);
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
            setNewlyCreatedItemId(null); // Clear highlight after move attempt
            console.log(`LOG: Clearing newlyCreatedItemId (${newlyCreatedItemId}) after successful/failed move attempt.`);
            console.log(`LOG: Clearing lastUploadedNames after successful/failed move attempt.`);
            setLastUploadedNames([]); // Also clear search list
            refreshInputData(); // Refresh data after move
            setSelectedNodes([]);
            treeRef.current?.deselectAll();
        } else {
            console.log("No changes detected from move operation, not refreshing.");
        }
    };
    // --- End Move Handler ---

    const handleCreateFolder = async () => {
        if (!newFolderName.trim() || isSubmittingModal) return;
        let parentPath = '.';
        let parentNode = null; // Keep track of the parent node
        // Target folder creation based on selection
        const selectedDir = selectedNodes.find(n => n.is_dir);
        if (selectedNodes.length === 1 && selectedDir) {
             parentPath = selectedDir.relativePath;
             parentNode = treeRef.current?.get(selectedDir.id); // Get the node itself
        } else if (selectedNodes.length > 0) {
             // If multiple items or a file is selected, find the parent of the first selected item
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
        } // else defaults to '.' (root)

        // --- START EDIT: Attempt to open the parent directory ---
        if (parentNode && parentNode.id && !parentNode.isOpen && treeRef.current) {
            try {
                console.log(`LOG: handleCreateFolder attempting to open parent node: ${parentNode.id}`);
                await treeRef.current.open(parentNode.id, { autoSelect: false });
                console.log(`LOG: handleCreateFolder successfully opened parent node: ${parentNode.id}`);
            } catch (openError) {
                 console.warn(`LOG: handleCreateFolder failed to open parent node ${parentNode.id}:`, openError);
                 // Continue anyway, maybe it doesn't exist yet or another issue
            }
        }
        // --- END EDIT ---

        const cleanParentPath = parentPath === '.' ? '' : parentPath.replace(/^\/+|\/+$/g, '');
        const cleanNewFolderName = newFolderName.trim().replace(/^\/+|\/+$/g, '');
        const fullPath = cleanParentPath ? `${cleanParentPath}/${cleanNewFolderName}` : cleanNewFolderName;
        
        setIsSubmittingModal(true);
        setModalError(null);
        try {
            await createInputDirectory(fullPath);
            setNewFolderName('');
            setIsCreateFolderModalOpen(false);
            // Set highlight target *before* refresh
            setNewlyCreatedItemId(fullPath);
            refreshInputData(); // Refresh data after creation
            // Trigger background activity briefly
            setBgIsActive(true);
            setTimeout(() => setBgIsActive(false), 1000);
        } catch (e) {
            console.error("Failed to create directory:", e);
            setModalError(e.message || 'Could not create folder.');
        } finally {
            setIsSubmittingModal(false);
        }
        setLastUploadedNames([]); // Also clear search list
    };

    const handleDownload = async () => {
        if (selectedNodes.length === 0 || isDownloading || triggerBgDownload) return; // Prevent overlap

        console.log(`LOG: Clearing newlyCreatedItemId (${newlyCreatedItemId}) due to handleDownload start.`);
        setNewlyCreatedItemId(null); // Clear highlight on download start
        setIsDownloading(true);
        setDownloadStatus({ message: `Starting download for ${selectedNodes.length} item(s)...`, error: false });
        console.log(`LOG: Clearing lastUploadedNames due to handleDownload start.`);
        setLastUploadedNames([]); // Also clear search list
        setTriggerBgDownload(true); // <-- Trigger background download animation

        let successCount = 0;
        let firstError = null;

        for (const node of selectedNodes) {
            try {
                setDownloadStatus({ message: `Downloading ${node.name}...`, error: false });
                const response = await downloadInputItem(node.relativePath);
                const blob = await response.blob();

                // Extract filename from content-disposition or use node name
                let filename = node.name;
                const disposition = response.headers.get('content-disposition');
                if (disposition && disposition.indexOf('attachment') !== -1) {
                    const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
                    const matches = filenameRegex.exec(disposition);
                    if (matches != null && matches[1]) {
                        filename = matches[1].replace(/['"]/g, '');
                    }
                }
                 // If it's a directory download, API likely sends a zip
                if (node.is_dir && !filename.toLowerCase().endsWith('.zip')) {
                    filename += '.zip';
                }

                // Create temporary link and click
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
                 // Small delay between downloads to prevent browser blocking
                 await new Promise(resolve => setTimeout(resolve, 200)); 
            } catch (error) {
                console.error(`Failed to download ${node.relativePath}:`, error);
                if (!firstError) {
                    firstError = error.message || `Failed to download ${node.name}.`;
                }
                 // Optionally break the loop on first error
                 // break; 
            }
        }

        setIsDownloading(false);
        if (firstError) {
            setDownloadStatus({ message: `Download failed: ${firstError}`, error: true });
        } else {
            setDownloadStatus({ message: `Successfully downloaded ${successCount} item(s).`, error: false });
        }
        // Clear status message after a delay
        setTimeout(() => setDownloadStatus({ message: '', error: false }), 5000);
    };

    const handleDelete = async () => {
        if (selectedNodes.length === 0 || isSubmittingModal || triggerBgDelete) return; // Prevent overlap

        const itemsToDelete = [...selectedNodes];
        setIsSubmittingModal(true);
        setModalError(null);
        setTriggerBgDelete(true); // <-- Trigger background delete animation
        setNewlyCreatedItemId(null); // Clear highlight on delete start
        console.log(`LOG: Clearing newlyCreatedItemId (${newlyCreatedItemId}) due to handleDelete start.`);
        console.log(`LOG: Clearing lastUploadedNames due to handleDelete start.`);
        setLastUploadedNames([]); // Also clear search list

        let successCount = 0;
        let firstError = null;

        const results = await Promise.allSettled(
            itemsToDelete.map(node => deleteInputItem(node.relativePath))
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

        if (successCount > 0) {
            // await fetchInitialData();
            refreshInputData(); // Refresh data after deletion
            setSelectedNodes([]);
            setBgIsActive(false); // Deactivate background animation after deletion
        }
    };

    const openCreateFolderModal = () => {
        setNewFolderName('');
        setModalError(null);
        console.log(`LOG: Clearing newlyCreatedItemId (${newlyCreatedItemId}) due to openCreateFolderModal.`);
        setIsCreateFolderModalOpen(true);
        setNewlyCreatedItemId(null); // Clear highlight when opening modal
        console.log(`LOG: Clearing lastUploadedNames due to openCreateFolderModal.`);
        setLastUploadedNames([]); // Also clear search list
        // Autofocus the input field
        setTimeout(() => createFolderInputRef.current?.focus(), 100);
    }

    const openDeleteModal = () => {
        setModalError(null);
        console.log(`LOG: Clearing newlyCreatedItemId (${newlyCreatedItemId}) due to openDeleteModal.`);
        setIsDeleteModalOpen(true);
        setNewlyCreatedItemId(null); // Clear highlight when opening modal
        console.log(`LOG: Clearing lastUploadedNames due to openDeleteModal.`);
        setLastUploadedNames([]); // Also clear search list
    }

    const onDrop = useCallback(
      async acceptedFiles => {
        console.log("LOG: onDrop START. Files dropped:", acceptedFiles.length);
        if (!acceptedFiles.length) return;
        
        // Clear previous highlight search list
        console.log("LOG: onDrop clearing lastUploadedNames at start.")
        setLastUploadedNames([]);

        // --- Record potential names for highlighting --- Start
        const namesToHighlight = acceptedFiles.map(file => {
            // If it's a directory upload (folder dragged in), webkitRelativePath exists
            if (file.webkitRelativePath) {
                // Extract the top-level folder name
                return file.webkitRelativePath.split('/')[0];
            } else {
                // Regular file upload
                return file.name;
            }
        });
        // Remove duplicates (important if a folder contains multiple files)
        const uniqueNames = [...new Set(namesToHighlight)];
        console.log("LOG: onDrop names recorded for potential highlight:", uniqueNames);
        setLastUploadedNames(uniqueNames);
        // --- Record potential names for highlighting --- End

        // Determine upload target based on selection
        let target = '.'; // Default to root
        let targetNode = null; // Track target node
        const selectedDir = selectedNodes.find(n => n.is_dir);
        if (selectedNodes.length === 1 && selectedDir) {
             target = selectedDir.relativePath;
             targetNode = treeRef.current?.get(selectedDir.id); // Get selected dir node
        } else if (selectedNodes.length > 0) {
             // If multiple items or a file is selected, upload to parent of first selected
             const firstNodeId = selectedNodes[0].id;
             const firstNode = treeRef.current?.get(firstNodeId);
             const parent = firstNode?.parent;
             if (parent) {
                 target = parent.data.relativePath;
                 targetNode = parent; // Target is the parent node
             } else {
                  target = '.'; // Default to root if no parent
                  targetNode = null;
             }
        }

        // --- START EDIT: Attempt to open the target directory ---
        if (targetNode && targetNode.id && !targetNode.isOpen && treeRef.current) {
             try {
                 console.log(`LOG: onDrop attempting to open target node: ${targetNode.id}`);
                 await treeRef.current.open(targetNode.id, { autoSelect: false });
                 console.log(`LOG: onDrop successfully opened target node: ${targetNode.id}`);
             } catch (openError) {
                 console.warn(`LOG: onDrop failed to open target node ${targetNode.id}:`, openError);
                 // Continue upload anyway
             }
        }
        // --- END EDIT ---

        console.log(`LOG: onDrop determined targetPath: ${target}`);
        console.log(`Uploading to target path: ${target}`);

        console.log("LOG: onDrop setting isUploading = true");
        setIsUploading(true);
        setUploadError(null);
        setLastUploadResult(null); // Clear previous success message
        console.log(`LOG: onDrop explicitly clearing newlyCreatedItemId (${newlyCreatedItemId}) at start of try block.`);
        setNewlyCreatedItemId(null); // Explicitly clear any previous highlight
        try {
          console.log(`LOG: onDrop BEFORE handleUpload call. Target: ${target}`);
          await handleUpload(acceptedFiles, target);
          console.log("LOG: onDrop AFTER handleUpload call completed.");
          // Highlight is now handled by the new useEffect hook based on lastUploadedNames
          console.log("LOG: onDrop setting lastUploadResult = true");
          setLastUploadResult(true); // Indicate success
          // Set names *and* trigger the search effect AFTER upload/refresh initiated
          console.log("LOG: onDrop setting lastUploadedNames and incrementing uploadCycle");
          setLastUploadedNames(uniqueNames); // Set names just before incrementing cycle
          setUploadCycle(c => c + 1); // Trigger the search effect
          // Clear success message after a delay
          setTimeout(() => {
              setLastUploadResult(null);
              // Maybe clear highlight here too, after a longer delay?
              // setNewlyCreatedItemId(null);
          }, 5000);
        } catch (e) {
          console.error('LOG: onDrop CAUGHT ERROR:', e);
          console.error('Upload failed:', e);
          setUploadError(e.message ?? 'An error occurred during upload.');
           // Clear error message after a delay
           setTimeout(() => setUploadError(null), 8000);
        } finally {
          console.log("LOG: onDrop setting isUploading = false (in finally block)");
          setIsUploading(false);
        }
      },
      [selectedNodes, handleUpload, treeRef] // Removed refreshInputData dependency here as it's called within handleUpload
    );
    
    const {
      getRootProps,
      getInputProps,
      isDragActive,
      open: openFileDialog,
    } = useDropzone({
      onDrop,
      noClick: true, // We have dedicated upload buttons
      noKeyboard: true,
      // Allow dropping directories (though handled via zipping in handleUpload)
      // Note: This doesn't directly enable folder uploads in all browsers without user selecting folder manually
      // accept: { 'application/zip': ['.zip'] }, // Primarily suggest zipping, but handleUpload adapts
    });

    const actionButtonClass = "flex items-center gap-2 px-3 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm transition-colors duration-200 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed";

    // Pass click tracking props to Node
    const handleNodeClick = (id, time) => {
        setLastClick({ id, time });
        // Set background active if a node is clicked (even if it becomes deselected on double click)
        setBgIsActive(true);
        console.log(`LOG: Clearing newlyCreatedItemId (${newlyCreatedItemId}) due to handleNodeClick.`);
        setNewlyCreatedItemId(null); // Clear highlight on click
        console.log(`LOG: Clearing lastUploadedNames due to handleNodeClick.`);
        setLastUploadedNames([]); // Also clear search list
         // Maybe reset after a short delay if only single click?
        // setTimeout(() => setBgIsActive(selectedNodes.length > 0), 400); // Check selection after double-click threshold
    };

    // Define the background click handler similar to ConfigSelector
    const handleBackgroundClick = useCallback((e) => {
        // Don't clear if a modal is open
        if (isDeleteModalOpen || isCreateFolderModalOpen) return;

        const clickedOnRow = e.target.closest('[role="treeitem"]');
        const clickedOnScrollbar = e.target.closest('.simplebar-scrollbar'); // Adjust if needed
        const clickedOnActionButton = e.target.closest('[data-action-button="true"]');
        const clickedOnModal = e.target.closest('[role="dialog"]'); // Check if click is inside any modal

        if (!clickedOnRow && !clickedOnScrollbar && !clickedOnActionButton && !clickedOnModal) {
            if (treeRef.current?.deselectAll) {
                treeRef.current.deselectAll();
            }
            if (typeof setSelectedNodes === 'function') {
                setSelectedNodes([]);
                setBgIsActive(false); // <-- Deactivate background on deselect all
                setNewlyCreatedItemId(null); // Clear highlight on background click
                console.log(`LOG: Clearing newlyCreatedItemId (${newlyCreatedItemId}) due to handleBackgroundClick.`);
                console.log(`LOG: Clearing lastUploadedNames due to handleBackgroundClick.`);
                setLastUploadedNames([]); // Also clear search list
            }
        }
    }, [isDeleteModalOpen, isCreateFolderModalOpen]);

    return (
        // Outer div for dropzone and background click handling, no padding here
        <div {...getRootProps({ className: 'flex flex-col h-full relative' })} onMouseDown={handleBackgroundClick}>
            <input
              {...getInputProps({
                multiple: true,
                // Enable directory uploads (may require zipping)
                webkitdirectory: "true",
                mozdirectory: "true",
                directory: "true",
              })}
            />

            {/* --- Add PourOverBackground --- */}
            <PourOverBackground
                isActive={bgIsActive}
                triggerDownload={triggerBgDownload}
                triggerDelete={triggerBgDelete}
                onDownloadComplete={handleDownloadAnimComplete}
                onDeleteComplete={handleDeleteAnimComplete}
            />

            {/* Main Content Area (Wrapped and Centered) */}
            <div className="relative z-10 flex-grow flex flex-col max-w-7xl w-full mx-auto p-6">

                {/* Header - Apply background, blur, padding, border */}
                <div className="flex justify-between items-center mb-4 bg-gray-900 bg-opacity-80 backdrop-blur-sm p-4 rounded-t-lg border-b border-gray-700">
                    <h1 className="text-2xl font-bold text-gray-100">Input Manager</h1>
                    <div className="flex items-center gap-2">
                        <EnhancedButton
                           onClick={openCreateFolderModal}
                           className={`${actionButtonClass} hover:bg-green-600 ${selectedNodes.length > 0 ? 'opacity-75' : ''}`}
                           title={selectedNodes.length === 1 && selectedNodes[0].is_dir ? `Create folder inside '${selectedNodes[0].name}'` : (selectedNodes.length > 0 ? `Create folder alongside selected item(s)` : 'Create folder in root')}
                           data-action-button="true"
                           particleColor="#22c55e"
                        >
                            <FiFolderPlus /> Create Folder
                        </EnhancedButton>
                        <EnhancedButton
                           onClick={openFileDialog}
                           className={`${actionButtonClass} hover:bg-indigo-600 ${selectedNodes.length > 0 ? 'opacity-75' : ''}`}
                           title={selectedNodes.length === 1 && selectedNodes[0].is_dir ? `Upload to '${selectedNodes[0].name}' (Folders will be zipped)` : (selectedNodes.length > 0 ? `Upload alongside selected item(s) (Folders will be zipped)`: `Upload to root (Folders will be zipped)`)}
                           data-action-button="true"
                           particleColor="#6366f1"
                        >
                            <FiUploadCloud /> Upload
                        </EnhancedButton>
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
                    <div className="flex-grow flex items-center justify-center text-gray-400 bg-gray-900 bg-opacity-80 backdrop-blur-sm p-4 rounded-b-lg">
                        <FiLoader className="animate-spin mr-3" /> Loading input files...
                    </div>
                )}
                {error && !treeData && (
                    <div className="flex-grow flex flex-col items-center justify-center text-red-400 bg-red-900 bg-opacity-60 backdrop-blur-sm p-6 rounded-b-lg border border-red-700">
                        <FiAlertTriangle className="w-10 h-10 mb-2" />
                        <p>Error loading input structure:</p>
                        <p className="text-sm mt-1">{error}</p>
                        <button onClick={() => fetchInputData(true)} className="mt-4 px-3 py-1.5 rounded bg-gray-600 hover:bg-gray-500 text-gray-200 text-sm">
                            Retry
                        </button>
                    </div>
                )}

                {/* Tree View Area - Apply background */}            
                {treeData && (
                     <div className={`flex-grow border-b border-r border-l border-gray-700 rounded-b-lg bg-gray-900 bg-opacity-80 backdrop-blur-sm overflow-auto relative min-h-0 ${isDragActive ? 'border-blue-500 ring-2 ring-blue-500 ring-opacity-50' : ''} ${isLoading ? 'pointer-events-none' : ''}`}>                    
                        {/* Dropzone overlay */}                 
                        {isDragActive && (
                            <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-70 z-10 pointer-events-none">
                                <p className="text-lg font-semibold text-blue-300">Drop files/folders here to upload</p>
                            </div>
                        )}
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
                                 height={1000} // Use large height, container handles scrolling
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
                                 children={(props) => <Node {...props} onNodeClick={handleNodeClick} lastClick={lastClick} newlyCreatedItemId={newlyCreatedItemId} />} 
                             >
                             </Tree>
                          ) : (
                              <div className="flex flex-col items-center justify-center h-full text-gray-500">
                                 <p className="mb-4">Input directory is empty.</p>
                                 <button onClick={openFileDialog} className={`${actionButtonClass} bg-blue-600 hover:bg-blue-500 text-white`}>
                                     <FiUploadCloud /> Upload Files or Folders
                                 </button>
                             </div>
                          )}
                    </div>
                )}

                {/* Status Bar Area - Apply background */}           
                <div className="h-8 mt-2 text-sm flex items-center gap-4 overflow-hidden whitespace-nowrap px-4 py-1 bg-gray-900 bg-opacity-80 backdrop-blur-sm rounded-lg border border-gray-700">
                    {/* Upload Status */}
                    {isUploading && (
                        <div className="flex items-center text-blue-400 flex-shrink-0">
                           <FiLoader className="animate-spin mr-2 flex-shrink-0" /> Uploading...
                        </div>
                    )}
                    {uploadError && <p className="text-red-400 truncate" title={uploadError}>Upload Error: {uploadError}</p>}
                    {lastUploadResult && <p className="text-green-400 flex items-center"><FiCheckCircle className="mr-1 flex-shrink-0"/> Upload successful!</p>}
                    
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
                {/* Modal-specific error display */} 
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
                    placeholder="e.g., my_data or path/to/my_data" 
                    className={`w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 mb-1
                                ${!newFolderName.trim() && isCreateFolderModalOpen ? 'animate-pulse-bg-subtle' : ''}`}
                    disabled={isSubmittingModal}
                    onKeyDown={(e) => e.key === 'Enter' && !isSubmittingModal && newFolderName.trim() && handleCreateFolder()} 
                />
                {/* Modal-specific error display */} 
                {modalError && <p className="text-red-400 text-xs mt-1 mb-3">Error: {modalError}</p>} 
                 <div className="flex justify-end gap-3 mt-4">
                    <button 
                       onClick={() => setIsCreateFolderModalOpen(false)} 
                       className="px-4 py-2 rounded bg-gray-600 hover:bg-gray-500 text-gray-200" 
                       disabled={isSubmittingModal}
                    >
                       Cancel
                    </button>
                    <EnhancedButton 
                       onClick={handleCreateFolder} 
                       className={`px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white flex items-center disabled:opacity-50
                                   ${newFolderName.trim() && !isSubmittingModal ? 'animate-pulse-blue-gold' : ''}`}
                       disabled={!newFolderName.trim() || isSubmittingModal}
                       particleColor="#60a5fa"
                    >
                        {isSubmittingModal ? <FiLoader className="animate-spin mr-2" /> : <FiFolderPlus className="mr-1"/>} 
                        {isSubmittingModal ? 'Creating...' : 'Create'}
                    </EnhancedButton>
                 </div>
            </Modal>

        </div>
    );
}

export default InputManager;
