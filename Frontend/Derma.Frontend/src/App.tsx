import './App.css'
import { Outlet } from 'react-router-dom'
import { Navbar } from './pages/Navbar'

function App() {
  

  return (
    <>
    <Navbar></Navbar>
      <Outlet></Outlet>
    </>
  )
}

export default App
