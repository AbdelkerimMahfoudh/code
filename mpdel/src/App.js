import React, { useState } from 'react';
import axios from 'axios';

function App() {
    const [formData, setFormData] = useState({
        country: '',
        goal: '',
    });

    const [prediction, setPrediction] = useState(null);

    // const handleChange = (e) => {
    //     setFormData({
    //         ...formData,
    //         [e.target.name]: e.target.value
    //     });
    // };

    // const handleSubmit = async (e) => {
    //     e.preventDefault();
    //     try {
    //         const response = await axios.post('http://127.0.0.1:5000/predict', formData);
    //         setPrediction(response.data.prediction);
    //     } catch (error) {
    //         console.error('Error making prediction:', error);
    //     }
    // };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input name="country" value={formData.country} onChange={handleChange} placeholder="Country" />
                <input name="goal" value={formData.goal} onChange={handleChange} placeholder="Goal" />
                <button type="submit">Predict</button>
            </form>
            {prediction !== null && (
                <div>
                    <h3>Prediction: {prediction}</h3>
                </div>
            )}
        </div>
    );
}

export default App;
