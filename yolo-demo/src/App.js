import React, { useState, useMemo } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Button } from './components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./components/ui/table";
import { LineChart, Line, Bar, BarChart, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Play } from 'lucide-react';




const YoloDemo = () => {
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedOptimization, setSelectedOptimization] = useState('');
  const [showDemo, setShowDemo] = useState(false);

  const models = ['YOLOv3', 'YOLOv4', 'YOLOv5'];
  const optimizations = ['None', 'Pruning', 'Quantization', 'Pruning + Quantization'];

  // Mock data for metrics
  const allMetricsData = {
    'YOLOv3': {
      'None': { mAP: 0.75, FLOPs: 65.7, ModelSize: 244, FPS: 30 },
      'Pruning': { mAP: 0.73, FLOPs: 50.2, ModelSize: 200, FPS: 35 },
      'Quantization': { mAP: 0.72, FLOPs: 65.7, ModelSize: 61, FPS: 40 },
      'Pruning + Quantization': { mAP: 0.70, FLOPs: 50.2, ModelSize: 50, FPS: 45 },
    },
    'YOLOv4': {
      'None': { mAP: 0.80, FLOPs: 70.5, ModelSize: 260, FPS: 28 },
      'Pruning': { mAP: 0.78, FLOPs: 55.0, ModelSize: 215, FPS: 33 },
      'Quantization': { mAP: 0.77, FLOPs: 70.5, ModelSize: 65, FPS: 38 },
      'Pruning + Quantization': { mAP: 0.75, FLOPs: 55.0, ModelSize: 54, FPS: 43 },
    },
    'YOLOv5': {
      'None': { mAP: 0.85, FLOPs: 75.2, ModelSize: 280, FPS: 26 },
      'Pruning': { mAP: 0.83, FLOPs: 60.0, ModelSize: 230, FPS: 31 },
      'Quantization': { mAP: 0.82, FLOPs: 75.2, ModelSize: 70, FPS: 36 },
      'Pruning + Quantization': { mAP: 0.80, FLOPs: 60.0, ModelSize: 58, FPS: 41 },
    },
  };


  return (
    <div className="container mx-auto p-4 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold mb-6 text-center text-blue-600">YOLO Model Demo</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <Select onValueChange={setSelectedModel}>
          <SelectTrigger>
            <SelectValue placeholder="Select YOLO version" />
          </SelectTrigger>
          <SelectContent>
            {models.map((model) => (
              <SelectItem key={model} value={model}>{model}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select onValueChange={setSelectedOptimization}>
          <SelectTrigger>
            <SelectValue placeholder="Select optimization method" />
          </SelectTrigger>
          <SelectContent>
            {optimizations.map((opt) => (
              <SelectItem key={opt} value={opt}>{opt}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {selectedModel && selectedOptimization && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Metrics for {selectedModel} with {selectedOptimization}</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={allMetricsData[selectedModel][selectedOptimization]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      <div className="text-center">
        <Button
          onClick={() => setShowDemo(!showDemo)}
          className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded"
        >
          <Play className="mr-2 h-4 w-4" /> Demonstration
        </Button>
      </div>

      {showDemo && (
        <div className="mt-6">
          <Card>
            <CardContent className="p-4">
              <div className="aspect-w-16 aspect-h-9 bg-gray-200 flex items-center justify-center">
                <p className="text-gray-500">Demo video would play here</p>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};


export default YoloDemo;