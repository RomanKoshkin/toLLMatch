import {createTheme} from '@mui/material/styles';

const themeObject = {
  palette: {
    background: {default: '#383838'},
    primary: {
      main: '#465A69',
    },
    info: {
      main: '#0064E0',
    },
    text: {primary: '#1C2A33'},
  },
  typography: {
    fontFamily: [
      'Optimistic Text',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {fontSize: '1rem', fontWeight: '500'},
  },
};

const theme = createTheme(themeObject);

/**
 * Set up a responsive font size at the 600px breakpoint
 */
// default is 1rem (16px)
theme.typography.body1[theme.breakpoints.down('sm')] = {fontSize: '0.875rem'};
// default is 1rem (16px)
theme.typography.h1[theme.breakpoints.down('sm')] = {fontSize: '0.875rem'};
// default is 0.875rem (14px)
theme.typography.button[theme.breakpoints.down('sm')] = {fontSize: '0.75rem'};
// default is 0.875rem (14px)
theme.typography.body2[theme.breakpoints.down('sm')] = {fontSize: '0.75rem'};

export default theme;
