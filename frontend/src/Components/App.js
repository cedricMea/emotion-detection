import React from 'react';
import InputForm from './InputForm'
import OutputResult from './OutputResult'
// import express from 'express'
import axios from 'axios'
import {Header} from 'semantic-ui-react'



class App extends React.Component {

  state = {
    reponse_data: {}
  }


  onInputApiFunc = async(endpoint, sentence) => {
    // this function fetch the flask api

    if (String(sentence).length !== 0){
      

      const api_link = `/api/${endpoint}?sentence=${sentence}`

      const reponse  = await axios.get(api_link)
      this.setState({reponse_data: reponse.data})
    }
  }

  render (){
    return (
        <div className="ui container" style={{marginTop:"70px"}}>
          <Header as='h1' textAlign='center' style={{marginBottom:"30px"}}> Sentiment detection App</Header>
          <InputForm apiFunc={this.onInputApiFunc}/>
          <OutputResult results={this.state.reponse_data} />
        </div>      
    )
  }
}


export default App;
