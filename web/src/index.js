'use strict';

//import dependencies
const express = require('express');
const path = require('path');
const exphbs = require('express-handlebars');
const cookieParser = require('cookie-parser');
const cors = require("cors");
require("dotenv").config({path: path.join(__dirname, "env/dev.env")})


//import internal files
const {route} = require('./app/routes');
// const db = require('./db');
const customMiddlewares = require('./utilities/middlewares/');


//init app
const app = express();


app.use(cors({
  origin: '*'
}));
//set static
app.use(express.static(path.join(__dirname, 'public')));


//set middleware
app.use(express.urlencoded({
  limit: '50mb',
  extended: true,
}));
app.use(express.json({ limit: '50mb' }));
app.use(cookieParser());
app.use(cors());



// custom middlewares
app.use(customMiddlewares);


//configure view engine
app.engine('hbs', exphbs({
  extname: 'hbs',
  layoutsDir: path.join(__dirname, 'views/layouts'),
  defaultLayout: 'main',
  helpers: require("./utilities/helpers"),
  partialsDir: [
    path.join(__dirname, 'views/components'),
  ]
}));
app.set('view engine', 'hbs');
app.set('views', path.join(__dirname, 'views'));



//connect database



//configure port
const port = process.env.PORT || 8080;



//routes
route(app);


app.listen(port, () => console.log(`App is running at http://127.0.0.1:${port}!`))