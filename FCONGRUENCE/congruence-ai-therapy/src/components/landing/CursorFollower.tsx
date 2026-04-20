import { useEffect, useState } from "react";
import { motion } from "framer-motion";

// Cursor follower component
export const CursorFollower = () => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);

  useEffect(() => {
    const updateMousePosition = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    const handleMouseEnter = () => setIsHovering(true);
    const handleMouseLeave = () => setIsHovering(false);

    window.addEventListener('mousemove', updateMousePosition);
    const interactiveElements = document.querySelectorAll('button, a, [role="button"], input');
    interactiveElements.forEach(el => {
      el.addEventListener('mouseenter', handleMouseEnter);
      el.addEventListener('mouseleave', handleMouseLeave);
    });
    return () => {
      window.removeEventListener('mousemove', updateMousePosition);
      interactiveElements.forEach(el => {
        el.removeEventListener('mouseenter', handleMouseEnter);
        el.removeEventListener('mouseleave', handleMouseLeave);
      });
    };
  }, []);

  return (
    <>
      <motion.div
        className="fixed top-0 left-0 w-3 h-3 bg-foreground/40 rounded-full pointer-events-none z-[90] mix-blend-difference"
        animate={{ x: mousePosition.x - 6, y: mousePosition.y - 6, scale: isHovering ? 2 : 1 }}
        transition={{ type: "spring", stiffness: 500, damping: 28 }}
      />
      <motion.div
        className="fixed top-0 left-0 w-8 h-8 border border-foreground/20 rounded-full pointer-events-none z-[90]"
        animate={{ x: mousePosition.x - 16, y: mousePosition.y - 16, scale: isHovering ? 1.5 : 1 }}
        transition={{ type: "spring", stiffness: 150, damping: 20 }}
      />
    </>
  );
};
