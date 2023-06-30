// get the client
const mysql = require("mysql2");

// Create the connection pool. The pool-specific settings are the defaults
const pool = mysql.createPool({
  host: process.env.DB_HOST || "localhost",
  user: process.env.DB_USER || "root",
  password: process.env.DB_PASSWORD || "Hien1987",
  database: process.env.DB_NAME || "test",
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  dateStrings: true,
});

const prom = pool.promise();

// create some function to utilize for all models
const db = {
  pool: prom,

  count: async (table = "table", conditions = "1=1" || ["1=1"]) => {
    // handling where statement from conditions
    let where = "";
    if (typeof conditions === typeof "a") {
      where = "where " + conditions;
    } else if (Array.isArray(conditions)) {
      where = "where " + conditions.length ? conditions.join(" and ") : "";
    } else {
      throw new Error("conditions must be a string or an string array");
    }


    const query = `select count(*) as 'count' from ${table} ${where}`;
    return await prom.execute(query);
  },

  get: async ( table = "table", fields = "*" || ["*"], conditions = "1=1" || ["1=1"], isPaging = false, page = 1, perPage = 8 ) => {
    // handling preventing negative number from limit statement
    page = (page >= 1 && typeof page === typeof 1) ? page : 1;
    perPage = (perPage >= 1 && typeof perPage === typeof 1) ? perPage : 8;
    const firstRow = (page - 1) * perPage;
    const limit = isPaging ? `limit ${firstRow},${perPage}` : ""

    // handling select statement from field
    let select = "*";
    if (typeof fields === typeof "a" || Array.isArray(fields)) {
      select = fields.toString();
    } else {
      throw new Error("Fiels must be a string or an array of string !!!");
    }

    // handling where statement from conditions
    let where = "";
    if (typeof conditions === typeof "a" && conditions !== "") {
      where = "where " + conditions;
    } else if (Array.isArray(conditions)) {
      where = "where " + conditions.length ? conditions.join(" and ") : "";
    } else {
      throw new Error("conditions must be a string or an string array");
    }

    const query = `select ${select} from ${table} ${where} ${limit}`;
    return await prom.execute(query);  
  },

  insert: async (table = "table", object = {}) => {
    const keyList = Object.keys(object).toString();
    const valueList = `"${Object.values(object).join('","')}"`;
    const query = `insert into ${table} (${keyList}) values (${valueList})`;
    return await prom.execute(query);
  },

  update: async (
    table = "table",
    object = {},
    conditions = "1=1" || ["1=1"]
  ) => {
    // handling where statement from conditions
    let where = "";
    if (typeof conditions === typeof "a" && conditions !== "") {
      where = "where " + conditions;
    } else if (Array.isArray(conditions)) {
      where = "where " + conditions.length ? conditions.join(" and ") : "";
    } else {
      throw new Error("conditions must be a string or an string array");
    }

    // handling set statement from conditions
    const keys = Object.keys(object);
    const values = Object.values(object);
    const set = `${keys.join("=?,")} =?`

    const query = `update ${table} set ${set} ${where} `;
    return await prom.execute(query, values);
  },
};

module.exports = db;
