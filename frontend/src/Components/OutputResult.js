import React from 'react';
import { Progress } from 'semantic-ui-react'
import { Grid, Header } from 'semantic-ui-react'
import Emojify from 'react-emojione';



class OutputResult extends React.Component {
    render () {
        //console.log(this.props.results)
        
        return(
            <div className="ui padded segment">
                <Header as='h2' textAlign='center'>
                    Outputs
                </Header>
                <Grid style={{marginTop:"30px"}}>

                    <OneClassOutput 
                        label="anger"
                        prediction={this.props.results.anger} 
                    />

                    <OneClassOutput 
                        label="joy"
                        prediction={this.props.results.joy} 
                    />

                    <OneClassOutput 
                        label="love"
                        prediction={this.props.results.love} 
                    />

                    <OneClassOutput 
                        label="sadness"
                        prediction={this.props.results.sadness} 
                    />
                </Grid>   
            </div>    
        )
    }
}



const OneClassOutput = (props) => {

    let prediction_percentage = props.prediction *100
    prediction_percentage =  prediction_percentage > 5 ? Math.round(prediction_percentage*10)/10 : Math.round(prediction_percentage)
    //const bar_color = (prediction_percentage>50) ? "green" : "grey"
    //<span> {props.label} </span>
    //<Progress percent={prediction_percentage} progress  color={bar_color} />

    const label_emoji_map = {
        "anger": "angry",
        "joy": "joy",
        "love": "heart",
        "sadness": "disappointed_relieved"
    }
    const emoji_text =  `:${label_emoji_map[props.label]}:`

    return(
        <Grid.Row>
            <Grid.Column width={12}>
                <Progress percent={prediction_percentage} progress color="blue" />

            </Grid.Column>

            <Grid.Column width={4}>
                {/* <span> {props.label} </span> */}
                <Emojify style={{color:"red"}}>
                    <span> {emoji_text} </span>
                </Emojify>
            </Grid.Column>
        </Grid.Row>  
    )
}



export default OutputResult