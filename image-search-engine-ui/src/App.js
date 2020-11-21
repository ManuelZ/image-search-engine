import React, { useState } from "react";

function App() {
  const [images, setImages] = useState([]);

  const handleFileChosen = async function (file) {
    const formData = new FormData();
    formData.append("image", file);

    const url = "http://127.0.0.1:5000/similar_images";
    let response = await fetch(url, {
      method: "POST",
      body: formData,
    });
    let data = await response.json();
    setImages(data["prediction"]);
  };

  const thumbnail = (im) => {
    return <img alt="" src={`data:image/jpeg;base64,${im}`} />;
  };

  /* TODO:
  https://github.com/react-dropzone/react-dropzone
  */

  let thumbnails = (
    <div className="m-10 grid grid-cols-6 gap-4">
      {images.map(([dist, im], i) => (
        <div className="flex flex-col items-center" key={i}>
          {thumbnail(im)}
          <span className="text-center">{Math.round(dist * 100) / 100}</span>
        </div>
      ))}
    </div>
  );

  return (
    <div className="App">
      <label className="font-bold">Select image</label>
      <input
        type="file"
        id="file"
        name="file"
        className=""
        accept=".jpg"
        onChange={(e) => handleFileChosen(e.target.files[0])}
      />
      {thumbnails}
    </div>
  );
}

export default App;
