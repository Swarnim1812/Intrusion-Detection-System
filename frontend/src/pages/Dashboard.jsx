/**
 * Enhanced Dashboard page with interactive charts and filters
 * Allows users to select metrics, time ranges, attack types, and chart types
 */

import { useState, useEffect } from 'react'
import ChartCard from '../components/ChartCard'
import Tooltip from '../components/Tooltip'
import { useToast } from '../utils/toastContext'
import { getVisualizationData, getMetrics } from '../services/api'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler,
} from 'chart.js'
import { Bar, Line, Pie, Doughnut } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  ChartTooltip,
  Legend,
  Filler
)

const Dashboard = () => {
  const [selectedMetric, setSelectedMetric] = useState('accuracy')
  const [selectedChartType, setSelectedChartType] = useState('line')
  const [dateRange, setDateRange] = useState('7d')
  const [selectedAttackType, setSelectedAttackType] = useState('all')
  const [chartData, setChartData] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [metrics, setMetrics] = useState(null)
  const { addToast } = useToast()
  const [attackFreq, setAttackFreq] = useState({ labels: [], values: [] })
  const [ratio, setRatio] = useState({ normal: 0, attack: 0 })
  const [attackTypes, setAttackTypes] = useState([]);

  useEffect(() => {
    const loadAttackTypes = async () => {
      try {
        const res = await fetch("http://localhost:5000/attack-types");
        const data = await res.json();
        setAttackTypes(data.attack_types || []);
      } catch (err) {
        console.error("Failed to load attack types:", err);
      }
    };
  
    loadAttackTypes();
  }, []);
  

  useEffect(() => {
    const loadRatio = async () => {
      try {
        const res = await fetch('http://localhost:5000/normal-attack-ratio')
        const data = await res.json()
        setRatio(data)
      } catch (err) {
        console.error("Failed to load normal-attack ratio:", err)
      }
    }
  
    loadRatio()
  }, [])
  

  useEffect(() => {
    const loadAttackFrequency = async () => {
      try {
        const res = await fetch('http://localhost:5000/attack-frequency')
        const data = await res.json()
        setAttackFreq(data)
      } catch (err) {
        console.error("Failed to load attack frequency", err)
      }
    }
  
    loadAttackFrequency()
  }, [])
  useEffect(() => {
    loadMetrics()
    updateChart()
  }, [selectedMetric, selectedChartType, dateRange, selectedAttackType])

  const loadMetrics = async () => {
    try {
      const data = await getMetrics()
      setMetrics(data)
    } catch (error) {
      console.error('Failed to load metrics:', error)
    }
  }

  const updateChart = async () => {
    setIsLoading(true)
    try {
      const result = await getVisualizationData({
        metric: selectedMetric,
        chart_type: selectedChartType,
        time_range: dateRange,
        attack_type: selectedAttackType,
      })
      setChartData(result.chart_data)
    } catch (error) {
      addToast('Failed to update chart', 'error')
      console.error(error)
    } finally {
      setIsLoading(false)
    }
  }

  const renderChart = () => {
    if (!chartData || isLoading) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      )
    }

    const commonOptions = {
      maintainAspectRatio: false,
      responsive: true,
    }

    switch (selectedChartType) {
      case 'bar':
        return <Bar data={chartData} options={commonOptions} />
      case 'line':
        return (
          <Line
            data={chartData}
            options={{
              ...commonOptions,
              scales: {
                y: {
                  beginAtZero: true,
                },
              },
            }}
          />
        )
      case 'pie':
        return <Pie data={chartData} options={commonOptions} />
      case 'doughnut':
        return <Doughnut data={chartData} options={commonOptions} />
      case 'heatmap':
        return (
          <Bar
            data={{
              labels: chartData?.labels || [],
              datasets: [
                {
                  label: "Normal",
                  data: chartData?.data?.[0] || [],
                  backgroundColor: "#10b981",
                },
                {
                  label: "Attack",
                  data: chartData?.data?.[1] || [],
                  backgroundColor: "#ef4444",
                },
              ],
            }}
            options={{
              ...commonOptions,
              scales: {
                x: { stacked: true },
                y: { stacked: true },
              },
            }}
          />
        )
        
      default:
        return <Bar data={chartData} options={commonOptions} />
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="mb-8" data-aos="fade-up">
        <h1 className="text-4xl font-bold mb-2">Analytics Dashboard</h1>
        <p className="text-gray-600">
          Interactive visualization of intrusion detection metrics and trends
        </p>
      </div>

      {/* Filters Section */}
      <div className="card mb-8" data-aos="fade-up">
        <h2 className="text-xl font-semibold mb-4">Chart Controls</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Metric Selector */}
          <div>
            <Tooltip text="Selecting this option will change which metric is displayed in the chart. Metrics include accuracy, precision, recall, and F1 score.">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Metric
              </label>
            </Tooltip>
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="input-field"
            >
              <option value="accuracy">Accuracy</option>
              <option value="precision">Precision</option>
              <option value="recall">Recall</option>
              <option value="f1">F1 Score</option>
            </select>
          </div>

          {/* Chart Type Selector */}
          <div>
            <Tooltip text="Selecting this option will change the type of graph displayed. Choose from bar, line, pie, doughnut, or heatmap charts.">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Chart Type
              </label>
            </Tooltip>
            <select
              value={selectedChartType}
              onChange={(e) => setSelectedChartType(e.target.value)}
              className="input-field"
            >
              <option value="line">Line Chart</option>
              <option value="bar">Bar Chart</option>
              <option value="pie">Pie Chart</option>
              <option value="heatmap">Heatmap</option>
            </select>
          </div>

          {/* Time Range Selector */}
          <div>
            <Tooltip text="This filter allows you to view data for different time periods. Select the range to analyze trends over time.">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Time Range
              </label>
            </Tooltip>
            <select
              value={dateRange}
              onChange={(e) => setDateRange(e.target.value)}
              className="input-field"
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
              {/* <option value="1y">Last year</option> */}
            </select>
          </div>

          {/* Attack Type Filter */}
          <div>
            <Tooltip text="This filter will show only the selected attack type. Choose 'All' to view all attack types together.">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Attack Type
              </label>
            </Tooltip>
            <select
              value={selectedAttackType}
              onChange={(e) => setSelectedAttackType(e.target.value)}
              className="input-field"
            >
              <option value="all">All Types</option>
              {attackTypes.map((type) => (
                <option key={type} value={type.toLowerCase()}>
                  {type}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
      {/* Main Chart */}
      <div className="mb-6" data-aos="fade-up" data-aos-delay="100">
        <ChartCard
          title={`${selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1)} Visualization`}
          description={`${selectedChartType.charAt(0).toUpperCase() + selectedChartType.slice(1)} chart showing ${selectedMetric} for ${dateRange}`}
        >
          {renderChart()}
        </ChartCard>
      </div>

      {/* Additional Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Attack Types Frequency */}
        <div data-aos="fade-up" data-aos-delay="200">
          <ChartCard
            title="Attack Types Frequency"
            description="Distribution of detected attack types"
          >
            <Bar
              data={{
                labels: attackFreq.labels,
                datasets: [{
                  label: "Frequency",
                  data: attackFreq.values,
                  backgroundColor: '#3b02f6',
                }]
              }}
              options={{
                maintainAspectRatio: false,
                responsive: true,
                plugins: {
                  legend: {
                    display: false,
                  },
                },
              }}
            />
          </ChartCard>
        </div>

        {/* Normal vs Attack Ratio */}
        <div data-aos="fade-up" data-aos-delay="300">
          <ChartCard
            title="Normal vs Attack Ratio"
            description="Overall classification distribution"
          >
            <Pie
              data={{
                labels: ['Normal', 'Attack'],
                datasets: [
                  {
                    data: [ratio.normal, ratio.attack],
                    backgroundColor: ['#10b981', '#ef4444'],
                  },
                ],
              }}
              options={{
                maintainAspectRatio: false,
                responsive: true,
              }}
            />
          </ChartCard>
        </div>
      </div>

      {/* Metrics Summary */}
      {metrics && (
        <div className="card" data-aos="fade-up" data-aos-delay="400">
          <h3 className="text-lg font-semibold mb-4">Current Model Metrics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">Accuracy</p>
              <p className="text-2xl font-bold text-primary-600">
                {((metrics.accuracy || 0) * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-600">Precision</p>
              <p className="text-2xl font-bold text-green-600">
                {((metrics.precision || 0) * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-600">Recall</p>
              <p className="text-2xl font-bold text-yellow-600">
                {((metrics.recall || 0) * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-600">F1 Score</p>
              <p className="text-2xl font-bold text-red-600">
                {((metrics.f1 || 0) * 100).toFixed(2)}%
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Dashboard
