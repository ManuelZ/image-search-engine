import React, { useState } from "react";
import config from "./config";

function App() {
  const [images, setImages] = useState([]);
  const [queryImage, setQueryImage] = useState(null);

  const handleFileChosen = async function (file) {
    setQueryImage(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append("image", file);
    let response = await fetch(config.ENDPOINT, {
      method: "POST",
      body: formData,
    });
    let data = await response.json();
    console.log(data)
    setImages(data["prediction"]);
  };

  /* TODO:
  https://github.com/react-dropzone/react-dropzone
  */

  let thumbnails = (
    <div className="mx-10 grid grid-cols-5 gap-4">
      {images.map(([dist, im, path], i) => (
        <div className="flex flex-col items-center" key={i}>
          <a href={`file://${path}`}>
            <img alt="" src={`data:image/jpeg;base64,${im}`} />
          </a>
          <span className="text-center">{Math.round(dist * 100) / 100}</span>
        </div>
      ))}
    </div>
  );

  return (
    <div className="flex flex-col">
      <div className="m-4">
        <label className="font-bold">Select image</label>
        <input
          type="file"
          id="file"
          name="file"
          className=""
          accept=".jpg"
          onChange={(e) => handleFileChosen(e.target.files[0])}
        />
      </div>

      <div className="flex flex-row ">
        <div className="flex flex-col mx-5 w-1/4 items-center justify-center">
          <img alt="" src={queryImage} />
          {queryImage ? (
            <p className="mt-1 text-sm text-gray-600 text-center">Query image</p>
          ) : (
            ""
          )}
        </div>
        <div className="flex w-3/4">{thumbnails}</div>
      </div>
    </div>
  );
}

export default App;
