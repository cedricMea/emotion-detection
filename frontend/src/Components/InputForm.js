import React from 'react';
import { Dropdown, Header, Form } from 'semantic-ui-react'

class InputForm extends React.Component {

    state = {
        selectedModelEndpoint: "",
        sentence: ""
    }
 
    modelOptions = [
        {
          key: 'largedatakey',
          text: 'Model with Large Data',
          value: 'largedatamodel'
          //image: { avatar: true, src: '/images/avatar/small/jenny.jpg' },
        },
        {
          key: 'customdatakey',
          text: 'Model with Custom Data',
          value: 'customdatamodel'
        }
    ]

    onFormSubmit = (event) => {
        event.preventDefault()
        this.props.apiFunc(this.state.selectedModelEndpoint, this.state.sentence) // fetch api
    }

    onDropdownChange = (event, {value}) => {
        
        this.setState({selectedModelEndpoint: value}) // change the state of selectedModelEndpoint
        this.props.apiFunc(value, this.state.sentence)  // fetch api. Je n'utilise pas le state car j'ai l'impression qu'il n'est pas mis a jour
    }



    render() {
        return(
            <div className="ui segment">
                <Header as='h2' textAlign='center'>
                Inputs
                </Header>
                <Form onSubmit={this.onFormSubmit}  action="" className="ui form">
                    <Form.Field>
                        <div className="ui massive icon input" style={{marginTop:"30px"}}>
                            <input 
                                type="text" placeholder='Enter a sentence' 
                                onChange={(event) => this.setState({sentence: event.target.value})}
                            />
                            <i className='write icon' ></i>
                        </div>
                    </Form.Field>
                    <Form.Field style={{marginTop:"30px", width:"30%"}}>
                        <Dropdown 
                            placeholder='Select your model'
                            fluid
                            selection
                            options={this.modelOptions}
                            onChange={this.onDropdownChange} 
                        />
                    </Form.Field>  
                </Form>
            </div>    
        )    
    }
}

export default  InputForm