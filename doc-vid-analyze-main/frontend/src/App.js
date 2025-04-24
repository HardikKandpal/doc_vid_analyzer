import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import Typography from '@mui/material/Typography';
import { AnimatePresence, motion } from 'framer-motion'; // Add this import

// Components
import Header from './components/Header';
import Footer from './components/Footer';
import ProtectedRoute from './components/ProtectedRoute';
import PayPalReturn from './components/PayPalReturn';

// Pages
import HomePage from './pages/HomePage';
import DocumentAnalyzerPage from './pages/DocumentAnalyzerPage';
import VideoAnalyzerPage from './pages/VideoAnalyzerPage';
import LegalChatbotPage from './pages/LegalChatbotPage';
import Auth from './pages/Auth';
import Subscription from './pages/Subscription';
import SubscriptionCallback from './pages/SubscriptionCallback';

// Services
import ApiService from './services/api';
import authService from './services/authService';

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#6200EA', // Purple from the gradient in your old UI
    },
    secondary: {
      main: '#03DAC6', // Teal from the gradient in your old UI
    },
    background: {
      default: '#f8f9fa',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 600,
    },
    h2: {
      fontWeight: 600,
    },
    h3: {
      fontWeight: 600,
    },
    button: {
      textTransform: 'none', // Prevents all-caps buttons
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.1)',
          },
          transition: 'all 0.2s ease-in-out',
        },
        containedPrimary: {
          background: 'linear-gradient(90deg, #6200EA 0%, #7c4dff 100%)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.05)',
          transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0px 8px 16px rgba(0, 0, 0, 0.1)',
          },
        },
      },
    },
  },
});

// Create a wrapper component that uses useLocation
function AnimatedRoutes({ userInfo, isAuthenticated, showNotification }) {
  const location = useLocation();
  
  // Animation variants for consistent transitions
  const pageTransition = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
    transition: { duration: 0.3, ease: "easeInOut" }
  };
  
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={
          <motion.div {...pageTransition}>
            <HomePage />
          </motion.div>
        } />
        
        <Route path="/login" element={!isAuthenticated ? 
          <motion.div {...pageTransition}>
            <Auth />
          </motion.div> 
          : <Navigate to="/" replace />} 
        />
        
        <Route path="/register" element={!isAuthenticated ? 
          <motion.div {...pageTransition}>
            <Auth initialTab={1} />
          </motion.div> 
          : <Navigate to="/" replace />} 
        />
        
        {/* Protected routes */}
        <Route 
          path="/document-analyzer" 
          element={
            <ProtectedRoute>
              <motion.div {...pageTransition}>
                <DocumentAnalyzerPage showNotification={showNotification} />
              </motion.div>
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/video-analyzer" 
          element={
            <ProtectedRoute>
              <motion.div {...pageTransition}>
                <VideoAnalyzerPage showNotification={showNotification} />
              </motion.div>
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/legal-chatbot" 
          element={
            <ProtectedRoute>
              <motion.div {...pageTransition}>
                <LegalChatbotPage showNotification={showNotification} />
              </motion.div>
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/subscription" 
          element={
            <ProtectedRoute>
              <motion.div {...pageTransition}>
                <Subscription showNotification={showNotification} />
              </motion.div>
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/subscription/callback" 
          element={
            <ProtectedRoute>
              <motion.div {...pageTransition}>
                <SubscriptionCallback showNotification={showNotification} />
              </motion.div>
            </ProtectedRoute>
          } 
        />
        
        {/* Fallback route */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AnimatePresence>
  );
}

function App() {
  const isAuthenticated = authService.isAuthenticated();
  const [loading, setLoading] = useState(true);
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'info'
  });
  // Remove the comment and use the state variable
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [userInfo, setUserInfo] = useState(null);

  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1000);

    // Check backend connection
    checkBackendConnection();

    // Get user info if authenticated
    if (isAuthenticated) {
      fetchUserInfo();
    }

    return () => clearTimeout(timer);
  }, [isAuthenticated]);

  // Update the checkBackendConnection function
  const checkBackendConnection = async () => {
    try {
      // Fix 2: Use ApiService instead of api
      const isConnected = await ApiService.checkHealth();
      setIsBackendConnected(isConnected);
      return isConnected;
    } catch (error) {
      console.error('Backend connection error:', error);
      setIsBackendConnected(false);
      return false;
    }
  };

  const fetchUserInfo = async () => {
    try {
      const result = await authService.getUserInfo();
      if (result.success) {
        setUserInfo(result.data);
      } else {
        // If we can't get user info, token might be invalid
        authService.logout();
      }
    } catch (error) {
      console.error('Failed to fetch user info:', error);
    }
  };

  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  const showNotification = (message, severity = 'info') => {
    setNotification({
      open: true,
      message,
      severity
    });
  };

  // Add a backend connection warning if not connected
  useEffect(() => {
    if (!isBackendConnected && !loading) {
      showNotification('Could not connect to backend server. Some features may not work.', 'warning');
    }
  }, [isBackendConnected, loading]);

  if (loading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box 
          sx={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: '100vh',
            background: 'linear-gradient(135deg, rgba(98, 0, 234, 0.05) 0%, rgba(3, 218, 198, 0.05) 100%)',
          }}
        >
          <Box 
            sx={{ 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center',
              animation: 'pulse 1.5s infinite ease-in-out',
              '@keyframes pulse': {
                '0%': { opacity: 0.6 },
                '50%': { opacity: 1 },
                '100%': { opacity: 0.6 }
              }
            }}
          >
            <CircularProgress size={60} thickness={4} />
            <Typography 
              variant="h6" 
              sx={{ 
                mt: 2, 
                background: 'linear-gradient(90deg, #6200EA 0%, #03DAC6 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                fontWeight: 600
              }}
            >
              Loading...
            </Typography>
          </Box>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <Header userInfo={userInfo} />
          <Box component="main" sx={{ flexGrow: 1, width: '100%' }}>
            {/* Use the AnimatedRoutes component instead of duplicating routes */}
            <AnimatedRoutes 
              userInfo={userInfo} 
              isAuthenticated={isAuthenticated} 
              showNotification={showNotification} 
            />
          </Box>
          <Footer />
        </Box>
      </Router>
      
      <Snackbar 
        open={notification.open} 
        autoHideDuration={6000} 
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notification.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

// Add this route to your Router
<Route path="/subscription/callback" element={<SubscriptionCallback />} />

export default App;
