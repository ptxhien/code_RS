# ERUDION
This is the title of 
**Project Link** - ***http://vvvshopadmin.herokuapp.com/***

**Features**

- Login & Logout



## Tech Stack 💻

- [Node.js](https://nodejs.org/en/)
- [Express.js](https://expressjs.com/)
- [Mysql](https://www.mysql.com/)


## Installation:

**1. Clone this repo by running the following command :**

```bash
 git clone https://github.com/yourname/your_repo
 cd ERUDION
```

**2. Install [Nodejs](https://nodejs.org/en/)**

**3. Install mysql:**

- Windows: ([Mysql installer](https://dev.mysql.com/downloads/installer/))
- Ubuntu: ([via bash](https://www.digitalocean.com/community/tutorials/how-to-install-mysql-on-ubuntu-20-04))
- Mac: ([via terminal](https://flaviocopes.com/mysql-how-to-install/))

**4. Import data:**

- [Import data](https://www.youtube.com/watch?v=uyP46E0UA9I) from file "../data/db/data_updated.sql" with whichever database you want


**5. Now install all the required packages by running the following commands :**

```
npm install
#or
yarn
```

**6. Create a .env file in env folder of src folder and add the following**

```
PORT=8080
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=yourpassword
DB_NAME=test
PRIVATE_KEY=zFUVn{;Sd4!]#lN
RS_API=http://127.0.0.1:6868

```


**7. Now start the node server by running the following command :-**

```
#Start the server
npm run start
#or
yarn start
```

**8.** **🎉 Open your browser and go to `http://127.0.0.1:8080`**

**9. If you prefer running in docker:-**
```
docker-compose up
```
