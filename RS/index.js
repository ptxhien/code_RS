var xhttp = new XMLHttpRequest();

xhttp.open('GET', 'http://127.0.0.1:6868/jobs', true);
xhttp.onload = function(){
    if (this.status == 200){
        // console.log(this.response);
        let data = JSON.parse(this.response);
        console(data)
    }
}
xhttp.send();