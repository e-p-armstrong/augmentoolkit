import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Placeholder SVGs (we can refine these later)
const FunnelSVG = () => (
    <svg width="600" height="300" viewBox="0 0 100 80" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M10 10 H90 L60 50 H40 L10 10 Z" stroke="#9CA3AF" strokeWidth="2" fill="#4B5563" /> {/* Funnel body */}
        <rect x="45" y="50" width="10" height="20" fill="#4B5563" stroke="#9CA3AF" strokeWidth="2" /> {/* Spout */}
        {/* Simple static "water" line */}
        <line x1="25" y1="30" x2="75" y2="30" stroke="#60A5FA" strokeWidth="3" />
    </svg>
);

const ContainerSVG = () => (
    <svg width="200" height="300" viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
        {/* Taller coffee level */}
        <rect x="10" y="35" width="40" height="35" fill="#A16207" /> {/* Coffee */}
        {/* Taller container outline */}
        <path d="M5 5 H55 V75 H5 V5 Z" stroke="#9CA3AF" strokeWidth="2" fill="#374151" /> {/* Glass */}
        {/* Handle */}
        <path d="M55 25 C 70 30, 70 50, 55 55" stroke="#9CA3AF" strokeWidth="4" fill="none" /> 
    </svg>
);

const SpillPuddleSVG = () => (
     <svg width="200" height="60" viewBox="0 0 100 30" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M5 15 Q 30 30, 50 15 T 95 20" stroke="#A16207" strokeWidth="4" fill="none" strokeLinecap="round"/>
    </svg>
);

const DripSVG = () => (
    <motion.svg
        width="50" height="50" viewBox="0 0 10 15" fill="none" xmlns="http://www.w3.org/2000/svg"
        style={{ originY: 0 }} // Animate scale from the top
    >
        <path d="M5 0 Q 0 5, 5 15 Q 10 5, 5 0 Z" fill="#A16207" /> {/* Water drip */}
    </motion.svg>
);

const LidSVG = () => (
    <svg width="130" height="50" viewBox="0 0 65 15" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="2.5" y="2.5" width="60" height="10" rx="3" fill="#4B5563" stroke="#9CA3AF" strokeWidth="2" />
    </svg>
);


const PourOverBackground = ({
    isActive = false,
    triggerDownload = false,
    triggerDelete = false,
    onDownloadComplete,
    onDeleteComplete,
}) => {
    const [containerState, setContainerState] = useState('idle'); // 'idle', 'downloading', 'spilled'

    // --- Animation Variants ---

    // Drip Animation
    const dripVariants = {
        idle: {
            y: [0, 130], // Start above container, end further down (Adjusted for scale)
            opacity: [1, 1, 0],
            scaleY: [0.5, 1, 1],
            transition: { duration: 1.5, repeat: Infinity, ease: "linear", times: [0, 0.8, 1] }
        },
        active: {
            // y: [0, 130], // Removed y animation for stationary effect
            y: 50,
            opacity: [0.6, 1, 0.6], // Pulsating opacity
            scaleY: [0.1, 4, 0.1], // Pulse scale from near zero to full height
            transition: { duration: 0.4, repeat: Infinity, ease: "easeInOut", times: [0, 0.5, 1] } // Symmetrical pulse timing
        },
        hidden: { opacity: 0, y: 0, transition: { duration: 0.1 } } // Hide drip during transitions
    };

    // Container Animation
    const containerVariants = {
        idle: { x: 0, rotate: 0, opacity: 1, transition: { duration: 0.5, ease: "easeOut" } },
        exitRight: { x: '150%', opacity: 0, transition: { duration: 0.8, ease: "easeIn" } },
        spill: { rotate: -90, x: -40, y: 20, transition: { duration: 0.4, ease: "easeIn" } }, // Tilt further left
        hidden: { opacity: 0, transition: { duration: 0.1 } }
    };

    // Lid Animation
    const lidVariants = {
        hidden: { y: -20, opacity: 0, transition: { duration: 0.3, ease: "easeOut" } },
        visible: { y: -5, opacity: 1, transition: { duration: 0.3, ease: "easeIn" } } // Position above container
    };

    // Spill Puddle Animation
    const spillPuddleVariants = {
        hidden: { opacity: 0, scale: 0.8, transition: { duration: 0.3, ease: "easeOut" } },
        visible: { opacity: 1, scale: 1, transition: { duration: 0.3, delay: 0.2, ease: "easeIn" } } // Appear slightly after spill starts
    };

    // --- Funnel Animation Variants ---
    const funnelVariants = {
        bobbing: {
            x: "-50%", // Maintain horizontal center (instead of relying on CSS translate)
            y: [-3, 3, -3], // Bob up and down slightly
            transition: { duration: 2.5, repeat: Infinity, ease: "easeInOut", repeatType: "loop" }
        }
    };

    // --- State Management ---

    // Handle Download Trigger
    useEffect(() => {
        if (triggerDownload) {
            setContainerState('downloading');
            // Sequence: Show lid -> Move container -> Hide lid -> Reset -> Callback
            const timer1 = setTimeout(() => { // Allow time for lid animation
                 const timer2 = setTimeout(() => { // Allow time for container exit
                    setContainerState('idle'); // Bring back a fresh container
                    onDownloadComplete?.();
                }, 800); // Duration of container exit
                 return () => clearTimeout(timer2);
            }, 300); // Duration of lid appear
             return () => clearTimeout(timer1);
        }
    }, [triggerDownload, onDownloadComplete]);

    // Handle Delete Trigger
    useEffect(() => {
        if (triggerDelete) {
            setContainerState('spilled');
            // Sequence: Spill -> Wait -> Reset -> Callback
            const timer1 = setTimeout(() => { // Let spill animation play
                setContainerState('idle'); // Bring back fresh container
                onDeleteComplete?.();
            }, 1500); // Duration to show spilled state
            return () => clearTimeout(timer1);
        }
    }, [triggerDelete, onDeleteComplete]);

    const showDrip = containerState === 'idle';
    const showLid = containerState === 'downloading';
    const showSpill = containerState === 'spilled';

    return (
        <div className="absolute inset-0 flex justify-center items-center overflow-hidden z-0 pointer-events-none">
            <div className="relative w-[300px] h-[440px]"> {/* Positioning context - Increased size */}

                {/* Funnel (Static for now) */}
                <motion.div
                    className="absolute top-0 left-[50%]" // Removed translate-x-1/2 as Framer Motion handles x now
                    initial={{ x: "-50%", y: -3 }} // Set initial position explicitly
                    variants={funnelVariants} // Apply variants
                    animate="bobbing"        // Start the animation
                 >
                    <FunnelSVG />
                </motion.div>

                {/* Drip */}
                 <AnimatePresence>
                    {showDrip && (
                         <motion.div
                            key={isActive ? "drip-active" : "drip-idle"}
                            className="absolute top-[200px] left-[50%] -translate-x-1/2 origin-top -z-10" // Position below funnel spout & set origin
                            style={{ translateX: '-25px' }} // Center the larger drip
                            variants={dripVariants}
                            initial="hidden"
                            animate={isActive ? 'active' : 'idle'}
                            exit="hidden"
                         >
                            <DripSVG />
                        </motion.div>
                    )}
                </AnimatePresence>


                {/* Container Group (Lid + Container) */}
                <motion.div
                    className="absolute top-[250px] left-[50%] -translate-x-1/2" // Position below funnel/drip (Adjusted for scale)
                    style={{
                        // Add origin for rotation if needed, default is center
                         originX: 0.5, 
                         originY: 0.5,
                         translateX: '-70px' // Adjust left slightly more
                    }}
                    variants={containerVariants}
                    initial="idle"
                    animate={
                        containerState === 'downloading' ? 'exitRight' :
                        containerState === 'spilled' ? 'spill' :
                        'idle'
                    }
                 >
                     {/* Lid */}
                     <motion.div
                        className="absolute top-[50px] left-[50%] -translate-x-1/2 z-10" // Position lid relative to container
                        style={{
                            translateX: '-90px'
                        }}
                        variants={lidVariants}
                        initial="hidden"
                        animate={showLid ? "visible" : "hidden"}
                    >
                        <LidSVG />
                    </motion.div>

                     {/* Container */}
                    <ContainerSVG />
                </motion.div>

                {/* Spill Puddle - Positioned independently below where container spills */}
                 <motion.div
                     className="absolute bottom-[-70px] left-[50%]" // Position roughly where spilled container base lands (Adjusted bottom)
                     style={{
                        translateX: '-300px'
                     }}
                     variants={spillPuddleVariants}
                     initial="hidden"
                     animate={showSpill ? "visible" : "hidden"}
                 >
                     <SpillPuddleSVG />
                 </motion.div>

            </div>
        </div>
    );
};

export default PourOverBackground;
