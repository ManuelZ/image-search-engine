import React, { useState } from "react";
import "./App.css";

function App() {
  const [images, setImages] = useState([]);

  const handleFileChosen = async function(file) {
    const formData = new FormData();
    formData.append("image", file);

    let params = {
      method: "POST",
      body: formData
    };

    const url = "http://127.0.0.1:5000/similar_images";
    let response = await fetch(url, params);
    let data = await response.json();
    console.log(data);
    setImages(data["prediction"]);
  };

  const thumbnail = im => {
    return <img src={`data:image/jpeg;base64,${im}`} />;
  };

  return (
    <div className="App">
      <label className="font-bold">Select image</label>
      <input
        type="file"
        id="file"
        name="file"
        className=""
        accept=".jpg"
        onChange={e => handleFileChosen(e.target.files[0])}
      />
      {images.map((pair, i) => (
        <div key={i}>
          {thumbnail(pair[1])}
          <span>{pair[0]}</span>
        </div>
      ))}
    </div>
  );
}

export default App;
