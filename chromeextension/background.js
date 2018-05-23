var xmlHttp = new XMLHttpRequest();

			xmlHttp.open("GET", "http://localhost:5000", true);
			//xmlHttp.onreadystatechange = handleServerResponse;
			console.log("reached server");
			xmlHttp.onload = function () {
			if (xmlHttp.readyState==4 && xmlHttp.status==200) {
			    console.log(xmlHttp.responseText);
        	}
};
			xmlHttp.send(null);


		
	


