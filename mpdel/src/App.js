import React, { useState } from 'react';
import axios from 'axios';

function App() {
    const [formData, setFormData] = useState({
        Project_name: '',
        country: '',
        backers_count: '',
        goal: '',
        pledged: '',
        Average_Contribution: ''
    });

    const [prediction, setPrediction] = useState(null);

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', formData);
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error('Error making prediction:', error);
        }
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input name="Project_name" value={formData.Project_name} onChange={handleChange} placeholder="Project Name" />
                <input name="country" value={formData.country} onChange={handleChange} placeholder="Country" />
                <input name="backers_count" value={formData.backers_count} onChange={handleChange} placeholder="Backers Count" />
                <input name="goal" value={formData.goal} onChange={handleChange} placeholder="Goal" />
                <input name="pledged" value={formData.pledged} onChange={handleChange} placeholder="Pledged" />
                <input name="Average_Contribution" value={formData.Average_Contribution} onChange={handleChange} placeholder="Average Contribution" />
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
