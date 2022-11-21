from transformers.models.bart.modeling_bart import *
import sys 
sys.path.append("..")
from summarization import (
    DualEncoderBartForConditionalGeneration,
    DualEncoderBartModel,
)

class BrioDualEncoderBartForConditionalGeneration(DualEncoderBartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BrioDualEncoderBartModel(config)



class BrioDualEncoderBartModel(DualEncoderBartModel):

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        memory_input_ids=None,
        memory_attention_mask=None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                memory_input_ids=memory_input_ids,
                memory_attention_mask=memory_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        if self.training:
            cand_num = decoder_input_ids.size(1)

            ## src
            encoder_hidden_states = encoder_outputs.src_last_hidden_state
            encoder_hidden_states = torch.repeat_interleave(encoder_hidden_states, cand_num, dim=0)
            attention_mask = torch.repeat_interleave(attention_mask, cand_num, dim=0)

            ## memory
            memory_hidden_states = encoder_outputs.memory_last_hidden_state
            memory_hidden_states = torch.repeat_interleave(memory_hidden_states, cand_num, dim=0)
            memory_attention_mask = torch.repeat_interleave(memory_attention_mask, cand_num, dim=0)

            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size(-1))
        else:
            encoder_hidden_states = encoder_outputs.src_last_hidden_state
            memory_hidden_states = encoder_outputs.memory_last_hidden_state
        

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            memory_hidden_states=memory_hidden_states,
            memory_attention_mask=memory_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.src_last_hidden_state,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )
    