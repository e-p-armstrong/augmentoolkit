import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, useAnimation, AnimatePresence } from 'framer-motion';

// Helper function for random numbers
const random = (min, max) => Math.random() * (max - min) + min;

/* ────────────────────────────────────────────────────────────────────────────
   Document: single paper in the stack
   ------------------------------------------------------------------------*/
const Document = ({
  id,
  yOffset,
  zIndex,
  onAnimationComplete,
  onSortMoveComplete,
  isTop,
  isBeingRemoved,
  isExploding,
  isSortingMove,
  targetYOffset,
  dimensions,
}) => {
  const controls = useAnimation();

  // Random explosion vectors (memoised per mount)
  const explodeX = useRef(random(-dimensions.width * 0.4, dimensions.width * 0.4)).current;
  const explodeY = useRef(random(-dimensions.height * 0.4, dimensions.height * 0.4)).current;
  const explodeRotate = useRef(random(-180, 180)).current;

  /* ---------------------------------------------------------------------
     Imperative animation orchestrator
  ---------------------------------------------------------------------*/
  const animateDocument = useCallback(async () => {
    if (isExploding) {
      // Radial blast
      await controls.start({
        x: explodeX,
        translateY: yOffset + explodeY,
        rotate: explodeRotate,
        opacity: 0,
        transition: { duration: 1.5, ease: 'easeOut' },
      });
    } else if (isBeingRemoved && isTop) {
      // Toss off-screen
      await controls.start({
        x: random(dimensions.width * 0.3, dimensions.width * 0.6) * (Math.random() > 0.5 ? 1 : -1),
        translateY: -dimensions.height * 0.6,
        rotate: random(-90, 90),
        opacity: 0,
        transition: { duration: 0.8, ease: 'easeOut' },
      });
      onAnimationComplete?.();
    } else if (isSortingMove) {
      /* --- Three-stage sideways shuffle --- */
      const sideOffset = 250;
      const moveOutDuration = 0.3;
      const moveDownDuration = 0.4;
      const moveInDuration = 0.3;

      // Bring to front instantly
      await controls.start({ opacity: 1, x: 0, translateY: yOffset, rotate: 0, zIndex: zIndex + 10 }, { duration: 0 });
      // 1. Sideways
      await controls.start({ x: sideOffset, transition: { duration: moveOutDuration, ease: 'easeOut' } });
      // 2. Downwards to slot
      await controls.start({ translateY: targetYOffset, transition: { duration: moveDownDuration, ease: 'circIn' } });
      // 3. Back in
      await controls.start({ x: 0, transition: { duration: moveInDuration, ease: 'easeOut' } });

      onSortMoveComplete?.(id);
    } else {
      // Idle re-stacking movement
      await controls.start({
        x: 0,
        translateY: yOffset,
        zIndex,
        rotate: 0,
        opacity: 1,
        transition: { duration: 0.5, ease: 'easeOut' },
      });
      onAnimationComplete?.();
    }
  }, [controls, dimensions, explodeRotate, explodeX, explodeY, id, isBeingRemoved, isExploding, isSortingMove, isTop, targetYOffset, yOffset, zIndex, onAnimationComplete, onSortMoveComplete]);

  /* ---------------------------------------------------------------------
     Run animator only *after* the motion.div has mounted.
  ---------------------------------------------------------------------*/
  useEffect(() => {
    const raf = requestAnimationFrame(animateDocument);
    return () => cancelAnimationFrame(raf);
  }, [animateDocument]);

  /* ---------------------------------------------------------------------
     Disable Framer‑Motion layout magic while this doc is under manual
     control (sorting, toss, or explosion). This prevents FM from measuring
     a fresh layout on every React re‑render – the culprit behind the
     "teleport‑back‑to‑centre" behaviour when the parent component updates.
  ---------------------------------------------------------------------*/
  const enableLayout = !isSortingMove && !isBeingRemoved && !isExploding;

  return (
    <motion.div
      key={id}
      {...(enableLayout ? { layout: 'position' } : {})}
      initial={{ x: 0, translateY: yOffset, zIndex, opacity: 1, rotate: 0 }}
      animate={controls}
      exit={{ opacity: 0, scale: 0.8, transition: { duration: 0.3 } }}
      className="absolute w-64 h-5 bg-gray-300 border border-gray-400 rounded-sm shadow-md origin-center"
    />
  );
};

/* ────────────────────────────────────────────────────────────────────────────
   DocumentStackBackground: orchestrates the pile of papers
   ------------------------------------------------------------------------*/
const DocumentStackBackground = ({
  triggerExplode = false,
  triggerTossTop = false,
  onExplodeComplete,
  onTossComplete,
}) => {
  const NUM_DOCUMENTS = 15;
  const STACK_OFFSET_Y = 30;

  const [documents, setDocuments] = useState([]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [isExploded, setIsExploded] = useState(false);

  const intervalRef = useRef(null);
  const [movingDocInfo, setMovingDocInfo] = useState({ id: null, targetYOffset: 0 });

  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const containerRef = useRef(null);

  /* -------------------------------------------------------------------
     Helpers: size, (re)initialisation, periodic shuffle
  -------------------------------------------------------------------*/
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        });
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  /* (re)build the stack */
  const initializeDocuments = useCallback(() => {
    if (dimensions.height === 0) return;

    const initialDocs = Array.from({ length: NUM_DOCUMENTS }).map((_, i) => ({
      id: `doc-${i}-${Date.now()}`,
      yOffset: (i - NUM_DOCUMENTS / 2 + 0.5) * STACK_OFFSET_Y,
      zIndex: i,
      isBeingRemoved: false,
    }));

    setDocuments(initialDocs);
    setMovingDocInfo({ id: null, targetYOffset: 0 });
    setIsExploded(false);
    setIsAnimating(false);
  }, [NUM_DOCUMENTS, STACK_OFFSET_Y, dimensions.height]);

  /* kick-off on mount */
  useEffect(() => initializeDocuments(), [initializeDocuments]);

  /* ---------------------------------------------------------------
     Idle shuffling logic (unchanged)
  ----------------------------------------------------------------*/
  const sortStep = useCallback(() => {
    if (isAnimating || documents.length < 2 || isExploded || movingDocInfo.id) return;

    const indexToRemove = Math.floor(random(1, documents.length - 1));
    const docToMove = documents[indexToRemove];
    if (!docToMove) return;

    const targetYOffset = (documents.length - 1 - documents.length / 2 + 0.5) * STACK_OFFSET_Y;

    setIsAnimating(true);
    setMovingDocInfo({ id: docToMove.id, targetYOffset });
  }, [STACK_OFFSET_Y, documents, isAnimating, isExploded, movingDocInfo.id]);

  /* … interval helpers (unchanged) … */
  const stopSortInterval = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const startSortInterval = useCallback(() => {
    stopSortInterval();
    if (!isAnimating && !isExploded) {
        intervalRef.current = setInterval(sortStep, 1000);
    }
  }, [sortStep, stopSortInterval, isAnimating, isExploded]);

  /* start-stop idle shuffle based on explosion and animation state */
  useEffect(() => {
    if (!isExploded && !isAnimating) {
        startSortInterval();
    } else {
        stopSortInterval();
    }
    return stopSortInterval;
  }, [isExploded, isAnimating, startSortInterval, stopSortInterval]);

  /* -----------------------------------------------------------------
     Toss-top trigger
     - Stop interval implicitly by setting isAnimating flag
     - Remove the doc from state *after* animation completes via onAnimationComplete
     - Interval restart is handled by the useEffect watching isAnimating
  ----------------------------------------------------------------*/
  useEffect(() => {
    if (triggerTossTop && documents.length > 0 && !isAnimating && !isExploded) {
      setIsAnimating(true);
      setDocuments((prev) => {
        const next = [...prev];
        if (next.length > 0) {
            next[next.length - 1] = { ...next[next.length - 1], isBeingRemoved: true };
        }
        return next;
      });
    }
  }, [triggerTossTop, documents.length, isAnimating, isExploded]);

  /* -----------------------------------------------------------------
     Explode trigger – *do not* immediately clear the list so that the
     papers have something to animate!
  ----------------------------------------------------------------*/
  useEffect(() => {
    if (triggerExplode && !isExploded && !isAnimating) {
      setIsAnimating(true);
      setIsExploded(true);
      setMovingDocInfo({ id: null, targetYOffset: 0 });
    }
  }, [triggerExplode, isExploded, isAnimating]);

  /* once exploded, wait, then rebuild the stack */
  useEffect(() => {
    if (!isExploded) return;

    const explosionDuration = 1500;
    const reinitDelay = 1000;
    const tid = setTimeout(() => {
      initializeDocuments();
      onExplodeComplete?.();
    }, explosionDuration + reinitDelay);

    return () => clearTimeout(tid);
  }, [isExploded, initializeDocuments, onExplodeComplete]);

  /* Callback for when *any* document's animation finishes */
  const handleDocumentAnimationComplete = useCallback((docId, isRemoval = false) => {
    if (isRemoval) {
        setDocuments((prev) => prev.filter(doc => doc.id !== docId));
        onTossComplete?.();
        setIsAnimating(false);
    }
  }, [onTossComplete]);

  /* -----------------------------------------------------------------*/
  return (
    <div
      ref={containerRef}
      data-id="document-stack-background"
      className="absolute inset-0 overflow-hidden z-0 pointer-events-none bg-gradient-to-b from-gray-800 via-gray-900 to-black"
    >
      <AnimatePresence>
        <div className="absolute inset-0 flex justify-center items-center">
          {documents.map((doc, idx) => (
            <Document
              key={doc.id}
              id={doc.id}
              yOffset={doc.yOffset}
              zIndex={doc.zIndex}
              isTop={idx === documents.length - 1}
              isBeingRemoved={doc.isBeingRemoved}
              isExploding={isExploded}
              isSortingMove={doc.id === movingDocInfo.id}
              targetYOffset={doc.id === movingDocInfo.id ? movingDocInfo.targetYOffset : 0}
              onAnimationComplete={() => handleDocumentAnimationComplete(doc.id, doc.isBeingRemoved)}
              onSortMoveComplete={(id) => {
                setDocuments((prev) => {
                  const i = prev.findIndex((d) => d.id === id);
                  if (i === -1) return prev;
                  const copy = [...prev];
                  const [moved] = copy.splice(i, 1);
                  copy.push(moved);
                  return copy.map((d, j) => ({
                    ...d,
                    yOffset: (j - copy.length / 2 + 0.5) * STACK_OFFSET_Y,
                    zIndex: j,
                  }));
                });
                setMovingDocInfo({ id: null, targetYOffset: 0 });
                setIsAnimating(false);
              }}
              dimensions={dimensions}
            />
          ))}
        </div>
      </AnimatePresence>
    </div>
  );
};

// Wrap the component with React.memo
const MemoizedDocumentStackBackground = React.memo(DocumentStackBackground);

export default MemoizedDocumentStackBackground;
