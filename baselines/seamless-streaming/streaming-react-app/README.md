# 🚀 Streaming React App

## Getting Started

This project uses the [Yarn Package Manager](https://yarnpkg.com/).

1. `yarn` - Install project dependencies
2. `yarn run dev` - Run the app with a development server that supports hot module reloading

NOTE: You will either need to provide the server URL via environment variable (you can use the `.env` file for this) or via a url param when you load the react app (example: `http://localhost:5173/?serverURL=localhost:8000`)

## URL Parameters

You can provide URL parameters in order to change the behavior of the app. Those are documented in [URLParams.ts](src/URLParams.ts).
