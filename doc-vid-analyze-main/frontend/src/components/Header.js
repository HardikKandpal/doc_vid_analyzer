import React, { useState } from 'react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Box,
  Avatar,
  Divider,
  useMediaQuery,
  useTheme,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import {
  Menu as MenuIcon,
  Person as PersonIcon,
  Description as DocumentIcon,
  Videocam as VideoIcon,
  QuestionAnswer as ChatIcon,
  CreditCard as SubscriptionIcon,
  Logout as LogoutIcon
} from '@mui/icons-material';
import authService from '../services/authService';

function Header({ userInfo }) {
  // Remove anchorEl state if not using it
  // const [anchorEl, setAnchorEl] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const isAuthenticated = authService.isAuthenticated();
  
  const handleLogout = () => {
    // handleMenuClose(); // Remove this if handleMenuClose is removed
    authService.logout();
    navigate('/login');
  };
  
  const toggleDrawer = (open) => (event) => {
    if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
      return;
    }
    setDrawerOpen(open);
  };
  
  const menuItems = [
    { text: 'Document Analyzer', icon: <DocumentIcon />, path: '/document-analyzer' },
    { text: 'Video Analyzer', icon: <VideoIcon />, path: '/video-analyzer' },
    { text: 'Legal Chatbot', icon: <ChatIcon />, path: '/legal-chatbot' },
    { text: 'Subscription', icon: <SubscriptionIcon />, path: '/subscription' },
  ];
  
  const drawer = (
    <Box
      sx={{ width: 250 }}
      role="presentation"
      onClick={toggleDrawer(false)}
      onKeyDown={toggleDrawer(false)}
    >
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" component="div">
          Legal Analyzer
        </Typography>
      </Box>
      <Divider />
      <List>
        <ListItem button component={RouterLink} to="/">
          <ListItemText primary="Home" />
        </ListItem>
        
        {isAuthenticated ? (
          <>
            {menuItems.map((item) => (
              <ListItem 
                button 
                key={item.text} 
                component={RouterLink} 
                to={item.path}
              >
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItem>
            ))}
            <Divider />
            <ListItem button onClick={handleLogout}>
              <ListItemIcon><LogoutIcon /></ListItemIcon>
              <ListItemText primary="Logout" />
            </ListItem>
          </>
        ) : (
          <ListItem button component={RouterLink} to="/login">
            <ListItemIcon><PersonIcon /></ListItemIcon>
            <ListItemText primary="Login" />
          </ListItem>
        )}
      </List>
    </Box>
  );
  
  return (
    <AppBar position="static">
      <Toolbar>
        {isMobile && (
          <IconButton
            size="large"
            edge="start"
            color="inherit"
            aria-label="menu"
            sx={{ mr: 2 }}
            onClick={toggleDrawer(true)}
          >
            <MenuIcon />
          </IconButton>
        )}
        
        <Typography
          variant="h6"
          component={RouterLink}
          to="/"
          sx={{
            flexGrow: 1,
            textDecoration: 'none',
            color: 'inherit',
            fontWeight: 700
          }}
        >
          Legal Document Analyzer
        </Typography>
        
        {!isMobile && (
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Button 
              color="inherit" 
              component={RouterLink} 
              to="/"
              sx={{ mx: 1 }}
            >
              Home
            </Button>
            
            {/* Always show menu items regardless of authentication */}
            {menuItems.map((item) => (
              <Button 
                key={item.text}
                color="inherit"
                component={RouterLink}
                to={item.path}
                sx={{ mx: 1 }}
              >
                {item.text}
              </Button>
            ))}
            
            {isAuthenticated ? (
              <IconButton
                // onClick={handleMenuOpen} // Remove this if handleMenuOpen is removed
                onClick={handleLogout} // Change to direct logout instead
                color="inherit"
                sx={{ ml: 2 }}
              >
                <Avatar sx={{ width: 32, height: 32, bgcolor: 'secondary.main' }}>
                  {userInfo?.email?.charAt(0).toUpperCase() || 'U'}
                </Avatar>
              </IconButton>
            ) : (
              <Button 
                color="inherit"
                component={RouterLink}
                to="/login"
                sx={{ ml: 1 }}
              >
                Login
              </Button>
            )}
          </Box>
        )}
      </Toolbar>
      
      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={toggleDrawer(false)}
      >
        {drawer}
      </Drawer>
    </AppBar>
  );
}

export default Header;