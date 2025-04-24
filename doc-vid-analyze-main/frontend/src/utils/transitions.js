import { keyframes } from '@emotion/react';

// Fade in animation
export const fadeIn = keyframes`
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
`;

// Slide up animation
export const slideUp = keyframes`
  from {
    transform: translateY(20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
`;

// Slide in from right
export const slideInRight = keyframes`
  from {
    transform: translateX(30px);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
`;

// Staggered animation helper
export const createStaggeredAnimation = (children, staggerDelay = 0.1) => {
  return children.map((child, index) => ({
    ...child,
    style: {
      ...child.props.style,
      animation: `${fadeIn} 0.5s ease forwards`,
      animationDelay: `${index * staggerDelay}s`,
      opacity: 0,
    },
  }));
};