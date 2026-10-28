import os
import json
import random
import re
from tqdm import tqdm
from prettytable import PrettyTable 
from termcolor import cprint
try:
    from pptree import Node, print_tree
except Exception:
    class Node:
        def __init__(self, name, parent=None):
            self.name, self.parent, self.children = name, parent, []
            if parent is not None:
                parent.children.append(self)

    def print_tree(node, *a, **k):
        def walk(n, d=0):
            print("  " * d + "- " + n.name)
            for c in getattr(n, "children", []):
                walk(c, d + 1)
        walk(node)

try:
    import google.generativeai as genai
except Exception:
    genai = None

from openai import OpenAI
import medgemma_backend as mg

# When False, answer options are presented in their labelled order (A, B, C, D)
# instead of upstream's shuffled order. Upstream shuffles the display order
# while keeping the letters attached, which smaller models can mis-map: they
# re-letter by position and give the right content under the wrong letter.
SHUFFLE_OPTIONS = True

# What happened inside the last question, recorded by main.py with the result.
# Upstream logs none of this, so a "multi-agent" answer could not be told apart
# from one where the debate never took place.
TRACE = {}

# --fixed: repair two upstream defects that stop MDAgents from behaving as a
# multi-agent system (see README, "Fixed variant"). False reproduces the
# published code exactly.
FIXED = False

class Agent:
    def __init__(self, instruction, role, examplers=None, model_info='gpt-4o-mini', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path

        if mg.is_medgemma(self.model_info):
            self.messages = [{"role": "system", "content": instruction}]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})
        elif self.model_info == 'gemini-pro':
            self.model = genai.GenerativeModel('gemini-pro')
            self._chat = self.model.start_chat(history=[])
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.client = OpenAI(api_key=os.environ['openai_api_key'])
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})

    def chat(self, message, img_path=None, chat_mode=True):
        if mg.is_medgemma(self.model_info):
            self.messages.append({"role": "user", "content": message})
            reply = mg.chat_completion(self.messages)
            self.messages.append({"role": "assistant", "content": reply})
            return reply

        if self.model_info == 'gemini-pro':
            for _ in range(10):
                try:
                    response = self._chat.send_message(message, stream=True)
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    return responses
                except:
                    continue
            return "Error: Failed to get response from Gemini."

        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.messages.append({"role": "user", "content": message})
            
            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            else:
                model_name = "gpt-4o-mini"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=self.messages
            )

            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content

    def temp_responses(self, message, img_path=None):
        if mg.is_medgemma(self.model_info):
            self.messages.append({"role": "user", "content": message})
            reply = mg.chat_completion(self.messages)
            self.messages.append({"role": "assistant", "content": reply})
            return {0.0: reply}

        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:      
            self.messages.append({"role": "user", "content": message})
            
            temperatures = [0.0]
            
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = 'gpt-4o-mini'
                response = self.client.chat.completions.create(
                    model=model_info,
                    messages=self.messages,
                    temperature=temperature,
                )
                
                responses[temperature] = response.choices[0].message.content
                
            return responses
        
        elif self.model_info == 'gemini-pro':
            response = self._chat.send_message(message, stream=True)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            return responses

class Group:
    def __init__(self, goal, members, question, examplers=None, model_info='gpt-4o-mini'):
        self.goal = goal
        self.members = []
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_info=model_info)
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]
            
            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {}".format(self.question)
            try:
                delivery = lead_member.chat(delivery_prompt)
            except:
                delivery = assist_members[0].chat(delivery_prompt)

            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery))
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])

            if self.examplers is not None:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, after reviewing the following example cases, return your answer to the medical query among the option provided:\n\n{self.examplers}\nQuestion: {self.question}"""
            else:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""

            response = lead_member.chat(investigation_prompt)

            return response

        elif comm_type == 'external':
            return

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy is None:
            hierarchy = 'Independent'
        
        # The model does not always write "parent > child"; anything else is
        # treated as reporting to the moderator. This only affects the printed
        # tree, not which agents take part or what they see.
        if 'independent' not in hierarchy.lower() and '>' in hierarchy:
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)

        count += 1

    return agents


def parse_recruited(recruited, max_agents=5):
    """
    Parse the recruiter's free-text roster into (role, description, hierarchy).

    Upstream assumed the model always answers in exactly the prompted format
    ("1. Cardiologist - Focuses on ... - Hierarchy: Independent") and indexed
    into split('-')[1] without guarding, so any deviation raised IndexError.
    Smaller models deviate often: markdown bullets, bold, colons instead of
    hyphens, or a lead-in sentence. We accept those shapes and skip anything
    that is clearly not an agent line.
    """
    import re

    agents = []
    for raw in (recruited or "").split('\n'):
        line = raw.strip()
        if not line:
            continue

        # Separate the optional "- Hierarchy: ..." tail.
        hierarchy = None
        m = re.split(r'\s*-\s*Hierarchy\s*:\s*', line, flags=re.I)
        if len(m) > 1:
            line, hierarchy = m[0].strip(), m[1].strip()

        # Strip list markers, numbering and markdown emphasis.
        line = re.sub(r'^\s*(?:[-*\u2022]|\d+[.)])\s*', '', line)
        line = line.replace('**', '').replace('__', '').strip()
        if not line:
            continue

        # Split role from description on the first ' - ' or ': '.
        parts = re.split(r'\s+-\s+|\s*:\s+', line, maxsplit=1)
        role = parts[0].strip().rstrip('.').lower()
        description = parts[1].strip().lower() if len(parts) > 1 else "contributes relevant medical expertise"

        # Reject obvious non-agent lines (prose, headers, over-long roles).
        if not role or len(role.split()) > 6 or role.endswith(('?', '!')):
            continue
        if any(role.startswith(w) for w in (
                'here', 'sure', 'okay', 'the following', 'experts', 'team',
                'note', 'answer', 'question', 'i ', 'we ')):
            continue

        agents.append((role, description, hierarchy))
        if len(agents) >= max_agents:
            break

    return agents


DEFAULT_ROSTER = [
    ("internist", "focuses on general internal medicine and differential diagnosis", None),
    ("clinical pharmacologist", "focuses on drug mechanisms and adverse effects", None),
    ("pathologist", "focuses on laboratory findings and disease mechanisms", None),
]

def parse_group_info(group_info):
    """
    Parse one MDT block from the recruiter's output.

    Guarded the same way as parse_recruited: upstream indexed into
    split(':')[1] and required every member line to start with "Member",
    which smaller models do not reliably produce. A group that ends up with
    no members is returned empty and skipped by the caller rather than
    crashing the run.
    """
    import re

    lines = (group_info or "").split('\n')
    parsed_info = {'group_goal': '', 'members': []}

    first = lines[0] if lines else ''
    goal_parts = first.split('-')[1:]
    parsed_info['group_goal'] = "".join(goal_parts).strip() or first.strip()

    for raw in lines[1:]:
        line = raw.strip().replace('**', '')
        if not line:
            continue
        # Accept "Member 1: Role - desc", "1. Role - desc", "- Role: desc"
        if re.match(r'^member\s*\d*\s*[:.]', line, flags=re.I):
            line = re.split(r'[:.]', line, maxsplit=1)[1].strip()
        elif re.match(r'^\s*(?:[-*\u2022]|\d+[.)])\s*', line):
            line = re.sub(r'^\s*(?:[-*\u2022]|\d+[.)])\s*', '', line).strip()
        else:
            continue
        if not line:
            continue

        parts = re.split(r'\s+-\s+|\s*:\s+', line, maxsplit=1)
        role = parts[0].strip()
        expertise = parts[1].strip() if len(parts) > 1 else 'contributes relevant medical expertise'
        if not role or len(role.split()) > 8:
            continue
        parsed_info['members'].append({
            'role': role,
            'expertise_description': expertise,
        })

    return parsed_info

def setup_model(model_name):
    if mg.is_medgemma(model_name):
        return None, mg.get_client()
    if 'gemini' in model_name:
        genai.configure(api_key=os.environ['genai_api_key'])
        return genai, None
    elif 'gpt' in model_name:
        client = OpenAI(api_key=os.environ['openai_api_key'])
        return None, client
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_data(dataset, data_root=None):
    root = data_root or os.environ.get('MDAGENTS_DATA', '../data')
    test_path = os.path.join(root, dataset, 'test.jsonl')
    train_path = os.path.join(root, dataset, 'train.jsonl')

    test_qa, examplers = [], []
    with open(test_path, 'r') as file:
        for line in file:
            if line.strip():
                test_qa.append(json.loads(line))

    if os.path.exists(train_path):
        with open(train_path, 'r') as file:
            for line in file:
                if line.strip():
                    examplers.append(json.loads(line))
    else:
        print(f"[warn] no {train_path}: running without few-shot examplers.")

    return test_qa, examplers

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sorted(sample['options'].items()):
            options.append("({}) {}".format(k, v))
        if SHUFFLE_OPTIONS:
            random.shuffle(options)
        question += " ".join(options)
        return question, None
    return sample['question'], None

def determine_difficulty(question, difficulty, model='gpt-3.5'):
    if difficulty != 'adaptive':
        return difficulty
    
    difficulty_prompt = f"""Now, given the medical query as below, you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) basic: a single medical agent can output an answer.\n2) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""
    
    medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info=model)
    medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.')
    response = medical_agent.chat(difficulty_prompt)
    TRACE['classifier_reply'] = response[:1000]

    if FIXED:
        return _fixed_difficulty(medical_agent, response)

    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic'
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate'
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced'
    print(f"[warn] no usable complexity label; defaulting to basic. Got: {response[:80]!r}")
    return 'basic'

_LABEL = re.compile(r'\b(basic|intermediate|advanced)\b')


def _fixed_difficulty(medical_agent, response):
    """Fix 1: route on the label the classifier chooses.

    Upstream returns 'basic' if the word "basic" or "1)" appears anywhere in
    the reply, so a reply that discusses the options is routed to the single
    agent whatever it concludes. Here the same agent, with its reasoning still
    in context, is asked to name its choice in one word. If that reply is
    unclear, the last label in the original reply is used.
    """
    choice = medical_agent.chat("State your final choice. Reply with exactly one word: "
                                "basic, intermediate, or advanced.")
    TRACE['classifier_choice_reply'] = choice[:200]
    named = set(_LABEL.findall(choice.lower()))
    if len(named) == 1:
        TRACE['classifier_parsed_from'] = 'choice'
        return named.pop()
    mentioned = _LABEL.findall(response.lower())
    if mentioned:
        TRACE['classifier_parsed_from'] = 'last_label'
        return mentioned[-1]
    TRACE['classifier_parsed_from'] = 'default'
    print(f"[warn] no usable complexity label; defaulting to basic. Got: {choice[:80]!r}")
    return 'basic'


def process_basic_query(question, examplers, model, args):
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)
    new_examplers = []
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            tmp_exampler = {}
            exampler_question = exampler['question']
            choices = [f"({k}) {v}" for k, v in sorted(exampler['options'].items())]
            if SHUFFLE_OPTIONS:
                random.shuffle(choices)
            exampler_question += " " + ' '.join(choices)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}\n\n"
            exampler_reason = medical_agent.chat(f"You are a helpful medical agent. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")

            tmp_exampler['question'] = exampler_question
            tmp_exampler['reason'] = exampler_reason
            tmp_exampler['answer'] = exampler_answer
            new_examplers.append(tmp_exampler)
    
    single_agent = Agent(instruction='You are a helpful assistant that answers multiple choice questions about medical knowledge.', role='medical expert', examplers=new_examplers, model_info=model)
    single_agent.chat('You are a helpful assistant that answers multiple choice questions about medical knowledge.')
    final_decision = single_agent.temp_responses(f'''The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\n**Question:** {question}\nAnswer: ''', img_path=None)
    
    return final_decision

def process_intermediate_query(question, examplers, model, args):
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = f"""You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."""
    
    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model)
    tmp_agent.chat(recruit_prompt)
    
    num_agents = 5  # You can adjust this number as needed
    recruited = tmp_agent.chat(f"Question: {question}\nYou can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nPlease answer in above format, and do not include your reason.")

    parsed = parse_recruited(recruited, num_agents)

    # Smaller models often ignore the requested format. Retry with a blunter
    # instruction before giving up, and report it: a run that silently fell
    # back to a default roster is not the same experiment.
    attempts = 0
    while len(parsed) < 2 and attempts < 2:
        attempts += 1
        cprint(f"[warn] recruiter output unparsable (attempt {attempts}); retrying", 'yellow')
        recruited = tmp_agent.chat(
            f"List exactly {num_agents} medical specialists for this question, "
            f"one per line, in this format and nothing else:\n"
            f"1. Role - one sentence description. - Hierarchy: Independent"
        )
        parsed = parse_recruited(recruited, num_agents)

    if len(parsed) < 2:
        cprint("[warn] recruiter still unparsable; using the default roster. "
               "This deviates from the protocol and must be reported.", 'red')
        parsed = list(DEFAULT_ROSTER)
        TRACE['default_roster'] = True

    TRACE['roster'] = [role for role, _, _ in parsed]

    # Rebuild the legacy shape so parse_hierarchy keeps working.
    agents_data = [(f"{i+1}. {role} - {desc}", hier)
                   for i, (role, desc, hier) in enumerate(parsed)]

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    agent_list = ""
    for i, (agent_role, description, _h) in enumerate(parsed):
        agent_list += f"Agent {i+1}: {agent_role} - {description}\n"

    agent_dict = {}
    medical_agents = []
    for agent_role, description, _h in parsed:
        inst_prompt = f"""You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."""
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model)
        
        _agent.chat(inst_prompt)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    for idx, (agent_role, description, _h) in enumerate(parsed):
        print(f"Agent {idx+1} ({agent_emoji[idx % len(agent_emoji)]} {agent_role}): {description}")

    fewshot_examplers = ""
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            exampler_question = f"[Example {ie+1}]\n" + exampler['question']
            options = [f"({k}) {v}" for k, v in sorted(exampler['options'].items())]
            if SHUFFLE_OPTIONS:
                random.shuffle(options)
            exampler_question += " " + " ".join(options)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
            exampler_reason = tmp_agent.chat(f"Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")
            
            exampler_question += f"\n{exampler_answer}\n{exampler_reason}\n\n"
            fewshot_examplers += exampler_question

    print()
    cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
    cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
    print_tree(hierarchy_agents[0], horizontal=False)
    print()

    num_rounds = 5
    num_turns = 5
    num_agents = len(medical_agents)

    interaction_log = {f'Round {round_num}': {f'Turn {turn_num}': {f'Agent {source_agent_num}': {f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}

    cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])

    round_opinions = {n: {} for n in range(1, num_rounds+1)}
    round_answers = {n: None for n in range(1, num_rounds+1)}
    initial_report = ""
    for k, v in agent_dict.items():
        opinion = v.chat(f'''Given the examplers, please return your answer to the medical query among the option provided.\n\n{fewshot_examplers}\n\nQuestion: {question}\n\nYour answer should be like below format.\n\nAnswer: ''', img_path=None)
        initial_report += f"({k.lower()}): {opinion}\n"
        round_opinions[1][k.lower()] = opinion

    final_answer = None
    for n in range(1, num_rounds+1):
        print(f"== Round {n} ==")
        round_name = f"Round {n}"
        agent_rs = Agent(instruction="You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.", role="medical assistant", model_info=model)
        agent_rs.chat("You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.")
        
        assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items())

        report = agent_rs.chat(f'''Here are some reports from different medical domain experts.\n\n{assessment}\n\nYou need to complete the following steps\n1. Take careful and comprehensive consideration of the following reports.\n2. Extract key knowledge from the following reports.\n3. Derive the comprehensive and summarized analysis based on the knowledge\n4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n\nYou should output in exactly the same format as: Key Knowledge:; Total Analysis:''')
        
        for turn_num in range(num_turns):
            turn_name = f"Turn {turn_num + 1}"
            print(f"|_{turn_name}")

            num_yes = 0
            for idx, v in enumerate(medical_agents):
                all_comments = "".join(f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n" for _k, _v in interaction_log[round_name][turn_name].items())
                
                participate = v.chat("Given the opinions from other medical experts in your team, please indicate whether you want to talk to any expert (yes/no)\n\nOpinions:\n{}".format(assessment if n == 1 else all_comments))
                TRACE.setdefault('talk_replies', []).append(
                    {'round': n, 'reply': participate[:300]})
                
                if 'yes' in participate.lower().strip():                
                    chosen_expert = v.chat(f"Enter the number of the expert you want to talk to:\n{agent_list}\nFor example, if you want to talk with Agent 1. Pediatrician, return just 1. If you want to talk with more than one expert, please return 1,2 and don't return the reasons.")
                    
                    chosen_experts = [int(ce) for ce in chosen_expert.replace('.', ',').split(',') if ce.strip().isdigit()]

                    for ce in chosen_experts:
                        specific_question = v.chat(f"Please remind your medical expertise and then leave your opinion to an expert you chose (Agent {ce}. {medical_agents[ce-1].role}). You should deliver your opinion once you are confident enough and in a way to convince other expert with a short reason.")
                        
                        print(f" Agent {idx+1} ({agent_emoji[idx]} {medical_agents[idx].role}) -> Agent {ce} ({agent_emoji[ce-1]} {medical_agents[ce-1].role}) : {specific_question}")
                        interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {ce}'] = specific_question
                
                    num_yes += 1
                else:
                    print(f" Agent {idx+1} ({agent_emoji[idx]} {v.role}): \U0001f910")

            if num_yes == 0:
                break
        
        if num_yes == 0:
            break

        tmp_final_answer = {}
        for i, agent in enumerate(medical_agents):
            response = agent.chat(f"Now that you've interacted with other medical experts, remind your expertise and the comments from other experts and make your final answer to the given question:\n{question}\nAnswer: ")
            tmp_final_answer[agent.role] = response

        round_answers[round_name] = tmp_final_answer
        final_answer = tmp_final_answer

    print('\nInteraction Log')        
    myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i]})" for i in range(len(medical_agents))])

    for i in range(1, len(medical_agents)+1):
        row = [f"Agent {i} ({agent_emoji[i-1]})"]
        for j in range(1, len(medical_agents)+1):
            if i == j:
                row.append('')
            else:
                i2j = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                j2i = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                
                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                elif i2j and j2i:
                    row.append(f'\u270B ({i}<->{j})')

        myTable.add_row(row)
        if i != len(medical_agents):
            myTable.add_row(['' for _ in range(len(medical_agents)+1)])
    
    print(myTable)

    # If no agent wanted to talk in round 1 the loop above exits before any
    # agent has given a final answer, so `final_answer` is still None and the
    # moderator below is asked to take a majority vote over "None".
    TRACE['rounds_with_answers'] = sum(v is not None for v in round_answers.values())
    TRACE['moderator_got_none'] = final_answer is None

    # Fix 2: when nobody asked to talk, the experts agreed after round 1, but
    # upstream then gives the moderator nothing. Pass on the answers they gave.
    if FIXED and final_answer is None:
        final_answer = dict(round_opinions[1])
        TRACE['moderator_got_none'] = False
        TRACE['moderator_given_round1'] = True

    cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])

    moderator = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model)
    moderator.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')
    
    _decision = moderator.temp_responses(f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote. Your answer should be like below format:\nAnswer: C) 2th pharyngeal arch\n{final_answer}\n\nQuestion: {question}", img_path=None)
    final_decision = {'majority': _decision}

    _mod_emoji = '\U0001F468\u200D\u2696\uFE0F'
    print(f"{_mod_emoji} moderator's final decision (by majority vote):", _decision)
    print()

    return final_decision

def process_advanced_query(question, model, args):
    print("[STEP 1] Recruitment")
    group_instances = []

    recruit_prompt = f"""You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."""

    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model)
    tmp_agent.chat(recruit_prompt)

    num_teams = 3  # You can adjust this number as needed
    num_agents = 3  # You can adjust this number as needed

    recruited = tmp_agent.chat(f"Question: {question}\n\nYou should organize {num_teams} MDTs with different specialties or purposes and each MDT should have {num_agents} clinicians. Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.\n\nFor example, the following can an example answer:\nGroup 1 - Initial Assessment Team (IAT)\nMember 1: Otolaryngologist (ENT Surgeon) (Lead) - Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.\nMember 2: General Surgeon - Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.\nMember 3: Anesthesiologist - Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.\n\nGroup 2 - Diagnostic Evidence Team (DET)\nMember 1: Endocrinologist (Lead) - Oversees the long-term management of Graves' disease, including hormonal therapy and monitoring for any related complications post-surgery.\nMember 2: Speech-Language Pathologist - Specializes in voice and swallowing disorders, providing rehabilitation services to improve the patient's speech and voice quality following nerve damage.\nMember 3: Neurologist - Assesses and advises on nerve damage and potential recovery strategies, contributing neurological expertise to the patient's care.\n\nGroup 3 - Patient History Team (PHT)\nMember 1: Psychiatrist or Psychologist (Lead) - Addresses any psychological impacts of the chronic disease and its treatments, including issues related to voice changes, self-esteem, and coping strategies.\nMember 2: Physical Therapist - Offers exercises and strategies to maintain physical health and potentially support vocal function recovery indirectly through overall well-being.\nMember 3: Vocational Therapist - Assists the patient in adapting to changes in voice, especially if their profession relies heavily on vocal communication, helping them find strategies to maintain their occupational roles.\n\nGroup 4 - Final Review and Decision Team (FRDT)\nMember 1: Senior Consultant from each specialty (Lead) - Provides overarching expertise and guidance in decision\nMember 2: Clinical Decision Specialist - Coordinates the different recommendations from the various teams and formulates a comprehensive treatment plan.\nMember 3: Advanced Diagnostic Support - Utilizes advanced diagnostic tools and techniques to confirm the exact extent and cause of nerve damage, aiding in the final decision.\n\nAbove is just an example, thus, you should organize your own unique MDTs but you should include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. When you return your answer, please strictly refer to the above format.")

    groups = [group.strip() for group in recruited.split("Group") if group.strip()]
    group_strings = ["Group " + group for group in groups]
    
    for i1, gs in enumerate(group_strings):
        res_gs = parse_group_info(gs)
        print(f"Group {i1+1} - {res_gs['group_goal']}")
        for i2, member in enumerate(res_gs['members']):
            print(f" Member {i2+1} ({member['role']}): {member['expertise_description']}")
        print()

        if not res_gs['members']:
            cprint(f"  [warn] group {i1+1} had no parsable members; skipping", 'yellow')
            continue
        group_instance = Group(res_gs['group_goal'], res_gs['members'], question, model_info=model)
        group_instances.append(group_instance)

    if not group_instances:
        cprint("[warn] no MDT groups could be parsed; using a single default "
               "team. This deviates from the protocol and must be reported.", 'red')
        TRACE['default_roster'] = True
        default_members = [{'role': r, 'expertise_description': d}
                           for r, d, _ in DEFAULT_ROSTER]
        group_instances = [Group('Initial Assessment Team (IAT)',
                                 default_members, question, model_info=model)]

    # STEP 2. initial assessment from each group
    # STEP 2.1. IAP Process
    initial_assessments = []
    for group_instance in group_instances:
        if 'initial' in group_instance.goal.lower() or 'iap' in group_instance.goal.lower():
            init_assessment = group_instance.interact(comm_type='internal')
            initial_assessments.append([group_instance.goal, init_assessment])

    initial_assessment_report = ""
    for idx, init_assess in enumerate(initial_assessments):
        initial_assessment_report += f"Group {idx+1} - {init_assess[0]}\n{init_assess[1]}\n\n"

    # STEP 2.2. other MDTs Process
    assessments = []
    for group_instance in group_instances:
        if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
            assessment = group_instance.interact(comm_type='internal')
            assessments.append([group_instance.goal, assessment])
    
    assessment_report = ""
    for idx, assess in enumerate(assessments):
        assessment_report += f"Group {idx+1} - {assess[0]}\n{assess[1]}\n\n"
    
    # STEP 2.3. FRDT Process
    final_decisions = []
    for group_instance in group_instances:
        if 'review' in group_instance.goal.lower() or 'decision' in group_instance.goal.lower() or 'frdt' in group_instance.goal.lower():
            decision = group_instance.interact(comm_type='internal')
            final_decisions.append([group_instance.goal, decision])
    
    compiled_report = ""
    for idx, decision in enumerate(final_decisions):
        compiled_report += f"Group {idx+1} - {decision[0]}\n{decision[1]}\n\n"

    # STEP 3. Final Decision
    decision_prompt = f"""You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query."""
    tmp_agent = Agent(instruction=decision_prompt, role='decision maker', model_info=model)
    tmp_agent.chat(decision_prompt)

    final_decision = tmp_agent.temp_responses(f"""Investigation:\n{initial_assessment_report}\n\nQuestion: {question}""", img_path=None)

    return final_decision